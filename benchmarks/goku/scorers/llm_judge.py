"""LLM-based scorers for Goku rubric types.

Implements 2 rubric types that require LLM judgement:
  - response_criteria: LLM judges whether a criterion is met
  - response_not_criteria: LLM judges whether a negative criterion is present
    (hallucination detection)
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path

import litellm

from benchmarks.goku.media_adapters import detect_provider, supports_native_pdf
from benchmarks.goku.media_render import pdf_to_page_images, video_to_keyframes
from benchmarks.goku.models import RubricItem, ScorerResult


logger = logging.getLogger(__name__)

LLM_JUDGE_TYPES = frozenset({"response_criteria", "response_not_criteria"})

# Supported media types for the multimodal judge payload.
_IMAGE_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
_PDF_SUFFIXES = {".pdf"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".avi", ".mkv"}

# Safety caps. Bedrock converse accepts up to ~20 images per request and a
# few MB per image; we stay well under both. Per-image cap is also a defense
# against accidentally encoding huge agent-output PNGs.
_MAX_MEDIA_PER_CALL = 20
_MAX_IMAGE_BYTES = 4_000_000   # 4 MB
_MAX_PDF_BYTES = 30_000_000    # 30 MB (Anthropic limit is 32 MB)
_KEYFRAMES_PER_VIDEO = 8       # mirrors run_infer.py agent path


def _image_url_block(path: Path) -> dict:
    """Encode an image file as an OpenAI-style image_url content block."""
    mime = _IMAGE_MIME_BY_SUFFIX.get(path.suffix.lower(), "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _build_media_blocks(
    paths: list[str] | list[Path] | None,
    *,
    judge_model: str = "",
    judge_canonical: str | None = None,
) -> tuple[list[dict], list[str]]:
    """Convert task input media paths into content blocks for the judge.

    Per-provider routing (mirrors the agent path in ``run_infer.py``):

      * Images (PNG/JPG/JPEG/GIF/WEBP) → ``image_url`` block (every provider).
      * PDFs:
          - Anthropic / Claude on Bedrock → native ``document`` block.
          - OpenAI / Gemini               → native ``file`` block.
          - Kimi via Bedrock (and any unknown) → render pages via pypdfium2
            and attach each page as an ``image_url`` block. This is the
            "ffmpeg/pypdfium2 fallback" the operator turns on by leaving the
            judge as Kimi.
      * Videos (MP4/MOV/WEBM/AVI/MKV) → uniformly extract keyframes via
        ffmpeg and attach as ``image_url`` blocks. Matches the agent path's
        uniform-ffmpeg decision so the judge sees what the agent saw.

    Returns:
        (blocks, warnings) — ``blocks`` is spliced into the message;
        ``warnings`` is appended to the judge's rationale so any silent
        drop (oversized file, render failure) is visible downstream.
    """
    blocks: list[dict] = []
    warnings: list[str] = []
    if not paths:
        return blocks, warnings

    provider = detect_provider(judge_model, judge_canonical)
    pdf_native = supports_native_pdf(judge_model, judge_canonical)

    def _cap_reached() -> bool:
        if len(blocks) >= _MAX_MEDIA_PER_CALL:
            msg = (
                f"judge media cap reached ({_MAX_MEDIA_PER_CALL}); "
                f"remaining inputs skipped"
            )
            logger.info(msg)
            warnings.append(msg)
            return True
        return False

    for p in paths:
        if _cap_reached():
            break

        path = Path(p)
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        size = path.stat().st_size

        # --- Images ---
        if suffix in _IMAGE_MIME_BY_SUFFIX:
            if size > _MAX_IMAGE_BYTES:
                msg = f"image {path.name} too large ({size} bytes) — skipping"
                logger.warning(msg)
                warnings.append(msg)
                continue
            blocks.append(_image_url_block(path))
            continue

        # --- PDFs ---
        if suffix in _PDF_SUFFIXES:
            if size > _MAX_PDF_BYTES:
                msg = f"PDF {path.name} too large ({size} bytes) — skipping"
                logger.warning(msg)
                warnings.append(msg)
                continue
            if pdf_native:
                b64 = base64.b64encode(path.read_bytes()).decode("ascii")
                if provider == "bedrock_anthropic":
                    blocks.append({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": b64,
                        },
                    })
                else:  # openai / gemini
                    blocks.append({
                        "type": "file",
                        "file": {
                            "filename": path.name,
                            "file_data": f"data:application/pdf;base64,{b64}",
                        },
                    })
            else:
                # Kimi-on-Bedrock or unknown — render pages to images.
                try:
                    pages = pdf_to_page_images(path, dpi=200)
                except Exception as exc:
                    msg = f"PDF render failed for {path.name}: {exc}"
                    logger.warning(msg)
                    warnings.append(msg)
                    continue
                for page in pages:
                    if _cap_reached():
                        break
                    blocks.append(_image_url_block(page))
            continue

        # --- Videos ---
        if suffix in _VIDEO_SUFFIXES:
            # Uniform ffmpeg approach: extract keyframes and attach as images.
            # Matches run_infer.py agent path so the judge sees the same
            # representation as the agent — clean apples-to-apples scoring.
            try:
                frames = video_to_keyframes(path, n_frames=_KEYFRAMES_PER_VIDEO)
            except Exception as exc:
                msg = f"video keyframe extract failed for {path.name}: {exc}"
                logger.warning(msg)
                warnings.append(msg)
                continue
            for frame in frames:
                if _cap_reached():
                    break
                blocks.append(_image_url_block(frame))
            continue

        # --- Anything else (unknown extension) ---
        msg = (
            f"input file {path.name} has no supported judge encoding "
            f"(extension {suffix!r}); skipping"
        )
        logger.info(msg)
        warnings.append(msg)

    return blocks, warnings

# Prompt template for response_criteria
# CRITICAL: the prompt MUST distinguish INPUT media (task fixture) from
# OUTPUT media (agent's work product). Conflating them leads to wrong
# verdicts on rubrics like "did the agent's output PDF correctly summarize
# the input image?" — the judge needs to look at the output PDF for the
# summary AND compare against the input image for correctness.
CRITERIA_PROMPT = """You are an evaluation judge for an AI agent benchmark.

## Task
Determine whether the following criterion is satisfied by the agent's output.
{media_note}
## Criterion
{criterion}

## Agent Text Response
{response}

## Agent Output Files (text-readable)
{file_contents}

## Agent Trajectory (for context)
{trajectory}

## Instructions
- Evaluate ONLY whether the criterion above is met.
- If the criterion is about what the AGENT PRODUCED (e.g. "the saved PDF
  contains a Conclusion section"), look at the OUTPUT media attachments.
- If the criterion is about what was in the TASK (e.g. "the agent
  identified all items visible in the photo"), look at the INPUT media
  attachments and compare against the agent's text response / output files.
- If the criterion involves both (e.g. "the agent's output image preserves
  features from the input image"), compare INPUT and OUTPUT side-by-side.
- The trajectory is provided for context only — do NOT evaluate the process,
  only the outcome.
- Be strict: the criterion must be clearly and fully satisfied.

## Required Output Format (JSON only, no other text)
{{"criteria_met": true/false, "reasoning": "brief explanation citing which media (INPUT or OUTPUT) you relied on"}}
"""

# Prompt template for response_not_criteria (hallucination detection)
NOT_CRITERIA_PROMPT = """You are an evaluation judge for an AI agent benchmark.

## Task
Determine whether the following NEGATIVE criterion (hallucination indicator)
is present in the agent's output.
{media_note}
## Negative Criterion (hallucination indicator)
{criterion}

## Agent Text Response
{response}

## Agent Output Files (text-readable)
{file_contents}

## Agent Trajectory (for context)
{trajectory}

## Instructions
- A hallucination is an assertion the agent makes (in its text response OR
  its output media) that does NOT appear in the INPUT media. Compare claims
  against the INPUT media attachments specifically.
- If the agent's OUTPUT media (e.g. a saved image) depicts something not
  present in the INPUT media, that is also a hallucination.
- "criteria_met": true means the hallucination IS present (BAD — penalty applies).
- "criteria_met": false means the hallucination is NOT present (GOOD — no penalty).
- A description that is INFERRED with appropriate hedging (e.g. "likely
  electrified given the modern sleeve") is NOT a hallucination — only direct
  factual claims about non-visible features qualify.
- Be thorough: check the text response, the output files, AND every attached
  INPUT media block before deciding.

## Required Output Format (JSON only, no other text)
{{"criteria_met": true/false, "reasoning": "brief explanation citing which INPUT media you compared against"}}
"""

# Single banner shown when ANY media is attached (input, output, or both).
# Specific per-section labels ("=== INPUT MEDIA ===" / "=== OUTPUT MEDIA ===")
# are inserted as separate text content blocks between the input and output
# attachments — that's more reliable than relying on the model to remember
# a banner instruction across many image blocks.
_MEDIA_NOTE_WITH = (
    "\nMedia attached to this message follows the text below. Sections are "
    "delimited by '=== INPUT MEDIA ===' (the task fixture given to the agent) "
    "and '=== OUTPUT MEDIA ===' (files the agent produced). Use each section "
    "for the appropriate part of the criterion — do not conflate them.\n"
)
_MEDIA_NOTE_WITHOUT = (
    "\n(This task has no attached media — judge using only the text below.)\n"
)


def score_llm_judge(
    item: RubricItem,
    response: str,
    file_contents: str,
    trajectory: str,
    judge_model: str = "bedrock/converse/moonshotai.kimi-k2.5",
    judge_api_key: str | None = None,
    judge_base_url: str | None = None,
    aws_region_name: str | None = None,
    input_image_paths: list[str] | list[Path] | None = None,
    output_media_paths: list[str] | list[Path] | None = None,
    judge_canonical_name: str | None = None,
) -> ScorerResult:
    """Score a rubric item using an LLM judge.

    Args:
        item: The rubric item to evaluate (response_criteria or response_not_criteria).
        response: The agent's final text response.
        file_contents: String representation of output file contents.
        trajectory: String representation of agent's action trajectory.
        judge_model: LiteLLM model identifier for the judge.
        judge_api_key: Optional API key override for the judge model.
            For Bedrock models, this is treated as AWS bearer token.
        judge_base_url: Optional base URL override for the judge model.
        aws_region_name: Optional AWS region for Bedrock models.
        input_image_paths: Optional list of TASK INPUT media paths (the
            fixture given to the agent). Per-provider routing: native PDF
            block where supported, ffmpeg keyframes for videos, image_url
            blocks for images. Pre-fix the judge had no input grounding and
            bluffed about visual claims.
        output_media_paths: Optional list of AGENT OUTPUT media paths (files
            the agent produced into its results/ directory). Attached with
            the SAME per-provider routing — the judge can natively inspect
            agent-produced PDFs/images/videos instead of seeing them as
            opaque "(binary, N bytes)" placeholders.
        judge_canonical_name: Optional canonical model name for Bedrock
            opaque-ARN models. Lets the multimodal router pick the right
            per-provider block shape.

    Returns:
        A ScorerResult with pass/fail based on LLM judgement.

    Raises:
        ValueError: If item.type is not an LLM judge type.
    """
    if item.type not in LLM_JUDGE_TYPES:
        raise ValueError(
            f"Rubric item #{item.number}: type '{item.type}' is not LLM-judged. "
            f"Expected one of: {sorted(LLM_JUDGE_TYPES)}"
        )

    # Select prompt template
    if item.type == "response_criteria":
        prompt_template = CRITERIA_PROMPT
    else:
        prompt_template = NOT_CRITERIA_PROMPT

    # Resolve media attachments first so the prompt's `{media_note}`
    # accurately reflects what the judge actually receives. Two independent
    # sections — INPUT media (task fixture) and OUTPUT media (agent's work
    # product) — must be clearly labeled in the payload so the judge can't
    # conflate them on rubrics that talk about both.
    input_blocks, input_warnings = _build_media_blocks(
        input_image_paths,
        judge_model=judge_model,
        judge_canonical=judge_canonical_name,
    )
    output_blocks, output_warnings = _build_media_blocks(
        output_media_paths,
        judge_model=judge_model,
        judge_canonical=judge_canonical_name,
    )
    has_any_media = bool(input_blocks or output_blocks)
    media_note = _MEDIA_NOTE_WITH if has_any_media else _MEDIA_NOTE_WITHOUT

    # Re-label warnings so the operator can tell which section dropped a
    # file (e.g. a corrupt OUTPUT video vs. an oversized INPUT image).
    media_warnings = (
        [f"INPUT: {w}" for w in input_warnings]
        + [f"OUTPUT: {w}" for w in output_warnings]
    )

    # Build the prompt
    prompt = prompt_template.format(
        media_note=media_note,
        criterion=item.criterion,
        response=response[:32000],
        file_contents=file_contents[:32000],
        trajectory=trajectory[:16000],
    )

    # Construct the message content.
    #
    # When media is present, build a multimodal content array with explicit
    # text delimiters between the INPUT and OUTPUT sections. The text
    # delimiters live INSIDE the content list (as separate text blocks) so
    # the judge sees them adjacent to the relevant image/document blocks —
    # more reliable than relying on the model to remember a banner from the
    # top of the message across many media blocks.
    #
    # When no media is present, fall back to the historical text-only shape
    # so providers that don't accept multimodal don't regress.
    if has_any_media:
        message_parts: list[dict] = [{"type": "text", "text": prompt}]
        if input_blocks:
            message_parts.append({
                "type": "text",
                "text": (
                    "\n=== INPUT MEDIA "
                    "(the task fixture given to the agent) ===\n"
                ),
            })
            message_parts.extend(input_blocks)
        if output_blocks:
            message_parts.append({
                "type": "text",
                "text": (
                    "\n=== OUTPUT MEDIA "
                    "(files the agent produced as its work product) ===\n"
                ),
            })
            message_parts.extend(output_blocks)
        message_content: str | list[dict] = message_parts
    else:
        message_content = prompt

    # Call LLM judge
    raw_content = ""
    judge_cost_usd = 0.0
    try:
        completion_kwargs: dict = {
            "model": judge_model,
            "messages": [{"role": "user", "content": message_content}],
            "temperature": 0.0,
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
        }
        if judge_base_url:
            completion_kwargs["base_url"] = judge_base_url

        is_bedrock = judge_model.startswith("bedrock/")
        key_str = str(judge_api_key) if judge_api_key else ""
        if hasattr(judge_api_key, "get_secret_value"):
            key_str = judge_api_key.get_secret_value()  # type: ignore[union-attr]
        if is_bedrock and key_str:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = key_str
            if aws_region_name:
                completion_kwargs["aws_region_name"] = aws_region_name
        elif key_str:
            completion_kwargs["api_key"] = key_str

        llm_response = litellm.completion(**completion_kwargs)
        raw_content = llm_response.choices[0].message.content or ""  # type: ignore[union-attr]

        # Capture cost from the response. LiteLLM stores it in
        # `_hidden_params["response_cost"]` when its pricing tables know the
        # model. Fall back to `litellm.completion_cost()` which computes from
        # token usage if not pre-attached. Either path failing leaves the
        # cost at 0 (no exception propagated — scoring must continue).
        try:
            hp = getattr(llm_response, "_hidden_params", None) or {}
            judge_cost_usd = float(hp.get("response_cost") or 0.0)
            if judge_cost_usd <= 0.0:
                judge_cost_usd = float(
                    litellm.completion_cost(completion_response=llm_response) or 0.0
                )
        except Exception:
            judge_cost_usd = 0.0

        # Strip markdown code fences if present
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[: -len("```")]
            cleaned = cleaned.strip()

        result = json.loads(cleaned)
        criteria_met = bool(result.get("criteria_met", False))
        reasoning = str(result.get("reasoning", "No reasoning provided"))

    except json.JSONDecodeError as e:
        logger.warning(
            "Judge returned invalid JSON for item #%d: %s. Raw: %s",
            item.number,
            e,
            raw_content[:200],
        )
        criteria_met = False
        reasoning = f"Judge returned invalid JSON: {raw_content[:200]}"
    except Exception as e:
        logger.exception("LLM judge call failed for item #%d", item.number)
        criteria_met = False
        reasoning = f"Judge call failed: {e}"

    # Map criteria_met to passed + points
    # For response_criteria: criteria_met=True → passed=True (positive)
    # For response_not_criteria: criteria_met=True → passed=True
    #   (hallucination detected → penalty applies)
    passed = criteria_met

    if item.points > 0:
        points_awarded = item.points if passed else 0
    else:
        # Negative items: penalty deducted when criterion IS matched
        points_awarded = item.points if passed else 0

    # If any media couldn't be routed to the judge, append a clearly-marked
    # note to the rationale so operators can spot blind-spot verdicts in
    # scores.jsonl. The judge's pass/fail stays whatever it returned —
    # we only annotate so the gap is visible, not silent.
    if media_warnings:
        reasoning = (
            reasoning
            + "\n\n[JUDGE MEDIA WARNINGS — judge did NOT see the following:]\n  - "
            + "\n  - ".join(media_warnings)
        )

    return ScorerResult(
        number=item.number,
        passed=passed,
        judge_rationale=reasoning,
        points_awarded=points_awarded,
        judge_cost_usd=judge_cost_usd,
    )
