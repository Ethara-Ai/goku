"""LLM-based scorers for Goku rubric types.

Implements 2 rubric types that require LLM judgement:
  - response_criteria: LLM judges whether a criterion is met
  - response_not_criteria: LLM judges whether a negative criterion is present
    (hallucination detection)
"""

from __future__ import annotations

import json
import logging
import os

import litellm

from benchmarks.goku.models import RubricItem, ScorerResult


logger = logging.getLogger(__name__)

LLM_JUDGE_TYPES = frozenset({"response_criteria", "response_not_criteria"})

# Prompt template for response_criteria
CRITERIA_PROMPT = """You are an evaluation judge for an AI agent benchmark.

## Task
Determine whether the following criterion is satisfied by the agent's output.

## Criterion
{criterion}

## Agent Response
{response}

## Agent Output Files
{file_contents}

## Agent Trajectory (for context)
{trajectory}

## Instructions
- Evaluate ONLY whether the criterion above is met
- Consider both the text response AND any output files
- The trajectory is provided for context only — do NOT evaluate the process, only the outcome
- Be strict: the criterion must be clearly and fully satisfied

## Required Output Format (JSON only, no other text)
{{"criteria_met": true/false, "reasoning": "brief explanation"}}
"""

# Prompt template for response_not_criteria (hallucination detection)
NOT_CRITERIA_PROMPT = """You are an evaluation judge for an AI agent benchmark.

## Task
Determine whether the following NEGATIVE criterion is present in the agent's output.
If present, the agent has hallucinated or produced incorrect information.

## Negative Criterion (hallucination indicator)
{criterion}

## Agent Response
{response}

## Agent Output Files
{file_contents}

## Agent Trajectory (for context)
{trajectory}

## Instructions
- Check if the agent's output contains the described hallucination or error
- "criteria_met": true means the hallucination IS present (BAD — penalty applies)
- "criteria_met": false means the hallucination is NOT present (GOOD — no penalty)
- Be thorough: check both text response and output files

## Required Output Format (JSON only, no other text)
{{"criteria_met": true/false, "reasoning": "brief explanation"}}
"""


def score_llm_judge(
    item: RubricItem,
    response: str,
    file_contents: str,
    trajectory: str,
    judge_model: str = "bedrock/converse/moonshotai.kimi-k2.5",
    judge_api_key: str | None = None,
    judge_base_url: str | None = None,
    aws_region_name: str | None = None,
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

    # Build the prompt
    prompt = prompt_template.format(
        criterion=item.criterion,
        response=response[:32000],
        file_contents=file_contents[:32000],
        trajectory=trajectory[:16000],
    )

    # Call LLM judge
    raw_content = ""
    judge_cost_usd = 0.0
    try:
        completion_kwargs: dict = {
            "model": judge_model,
            "messages": [{"role": "user", "content": prompt}],
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

    return ScorerResult(
        number=item.number,
        passed=passed,
        judge_rationale=reasoning,
        points_awarded=points_awarded,
        judge_cost_usd=judge_cost_usd,
    )
