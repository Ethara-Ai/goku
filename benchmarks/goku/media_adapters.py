"""Per-provider native PDF content-block builders for the AGENT path.

The three agent models we run (Claude Opus 4.7 on Bedrock, GPT-5.5 on OpenAI,
Gemini 3.1 Pro on Google) all accept native PDF input via the chat API — but
each expects a *different* content-block shape. This module routes a PDF to
the right shape based on the LiteLLM model identifier.

Empirically verified shapes (see ``/tmp/multimodal_probe/`` probe results):

  * Claude on Bedrock  → ``{"type": "document", "source": {...}}``  (Anthropic-style)
  * GPT-5.5 on OpenAI  → ``{"type": "file", "file": {"filename", "file_data"}}``  (OpenAI-style)
  * Gemini direct      → same OpenAI-style ``file`` block (Gemini accepts it)

Videos are uniformly handled by extracting keyframes (see ``media_render``)
because only Gemini natively supports video via API — going asymmetric across
the 3 agent models would muddy the eval. Keyframe images go through whatever
the model's regular image-block shape is.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


# Substrings to match against the LiteLLM model string. Order matters only
# for disambiguation; in practice these are mutually exclusive.
_BEDROCK_ANTHROPIC_MARKERS = ("anthropic.claude", "claude-opus", "claude-sonnet", "claude-haiku")
_GEMINI_MARKERS = ("gemini",)
_OPENAI_MARKERS = ("openai/", "gpt-5", "gpt-4", "o1-", "o3-", "o4-")


def detect_provider(
    model_string: str,
    model_canonical_name: str | None = None,
) -> str:
    """Return one of: 'bedrock_anthropic', 'gemini', 'openai', 'bedrock_kimi', 'unknown'.

    Bedrock application-inference-profile ARNs are opaque (e.g.
    ``bedrock/converse/arn:aws:bedrock:.../application-inference-profile/abc123``)
    and contain none of the provider markers we'd otherwise key on. Callers
    can pass ``model_canonical_name`` (e.g. ``anthropic.claude-opus-4-7``) so
    we can still detect the underlying model family — this attribute is set
    on our LLM configs precisely for that purpose.
    """
    haystack = (model_string + " " + (model_canonical_name or "")).lower()
    if "kimi" in haystack or "moonshotai" in haystack:
        return "bedrock_kimi"
    if "gemini" in haystack:
        return "gemini"
    if any(s in haystack for s in _BEDROCK_ANTHROPIC_MARKERS) or "anthropic" in haystack:
        return "bedrock_anthropic"
    if any(s in haystack for s in _OPENAI_MARKERS):
        return "openai"
    return "unknown"


def build_pdf_block(
    pdf_path: str | Path,
    model_string: str,
    model_canonical_name: str | None = None,
) -> dict:
    """Return the ONE content block dict that natively encodes the PDF for ``model_string``.

    For opaque Bedrock ARNs, pass ``model_canonical_name`` so the underlying
    provider can be detected (the ARN itself carries no provider markers).

    Raises ``NotImplementedError`` for providers that don't accept native PDF
    (currently Kimi via Bedrock) — caller should fall back to rendering pages
    to images via :mod:`benchmarks.goku.media_render`.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    provider = detect_provider(model_string, model_canonical_name)
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")

    if provider == "bedrock_anthropic":
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": b64,
            },
        }
    if provider in ("openai", "gemini"):
        return {
            "type": "file",
            "file": {
                "filename": path.name,
                "file_data": f"data:application/pdf;base64,{b64}",
            },
        }
    if provider == "bedrock_kimi":
        raise NotImplementedError(
            "Kimi K2.5 via Bedrock Converse does not support PDF input. "
            "Fall back to rendering pages to images via "
            "benchmarks.goku.media_render.pdf_to_page_images()."
        )
    raise ValueError(
        f"No PDF block builder for provider {provider!r} (model={model_string!r})"
    )


def supports_native_pdf(
    model_string: str, model_canonical_name: str | None = None
) -> bool:
    """True if the model's API exposes native PDF input."""
    return detect_provider(model_string, model_canonical_name) in {
        "bedrock_anthropic", "openai", "gemini",
    }


def supports_native_video(
    model_string: str, model_canonical_name: str | None = None
) -> bool:
    """True if the model's API exposes native video input.

    Per empirical probe: only Gemini direct supports video via the chat API.
    (Kimi-the-model has video capability but Bedrock doesn't expose it; GPT-5.5
    and Claude Opus 4.7 reject video input outright.)
    """
    return detect_provider(model_string, model_canonical_name) == "gemini"
