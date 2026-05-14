"""
Goku benchmark configuration.

Default values for the Goku multimodal agentic evaluation benchmark.
"""

# Inference defaults (used by run_infer.py)
INFER_DEFAULTS = {
    "tasks_dir": "tasks",
    "num_workers": 10,
    "runs_per_model": 3,
    "task_timeout_seconds": 600,  # 10 minutes per task run
    "max_retries": 2,
    "enable_condenser": True,
    "condenser_max_size": 240,
    "condenser_keep_first": 2,
    "judge_temperature": 0.0,
    "judge_max_tokens": 2048,
}

# Calibration thresholds
CALIBRATION_DEFAULTS = {
    "too_easy_threshold": 0.9,  # All models above this = too easy
    "too_hard_threshold": 0.3,  # All models below this = too hard
    "target_min_score": 0.7,  # At least 1 model should exceed this
}

# Display names for models in the delivery folder structure.
# Maps short aliases (exact match) and long-form slug substrings to clean
# delivery names per the doc spec (e.g. "claude-opus", "gpt5.5", "gemini-3.1").
#
# Lookup order in get_model_display_name():
#   1) Exact (case-insensitive) match — handles short aliases passed via --models
#      such as "bedrock_converse_arn", "openai", "gemini".
#   2) Substring match — handles long output-dir slugs such as
#      "gemini_gemini-3.1-pro-preview_sdk_...".
# Exact match runs first so a short alias never gets accidentally captured by a
# too-greedy substring pattern. This table is consumed ONLY by the delivery
# packager (eval_infer.export_delivery_format) for folder naming — it has no
# effect on inference, LiteLLM calls, or Bedrock routing.
MODEL_DISPLAY_NAMES: dict[str, str] = {
    # Short aliases (intended for exact match)
    "bedrock_converse_arn": "claude-opus",
    "openai": "gpt5.5",
    "gemini": "gemini-3.1",
    # Long-form substrings (intended for substring match against output paths)
    "claude-opus": "claude-opus",
    "opus-4": "claude-opus",
    "653flds7ip4s": "claude-opus",
    "gpt-5": "gpt5.5",
    "gpt5": "gpt5.5",
    "gemini-3": "gemini-3.1",
    "gemini_gemini": "gemini-3.1",
    "kimi": "kimi-k2.5",
}


def get_model_display_name(model_id: str) -> str:
    """Return a clean display name for a model id or slug.

    Resolution order:
      1) Exact (case-insensitive) match against MODEL_DISPLAY_NAMES keys.
      2) Case-insensitive substring match (first matching key wins).
      3) Fallback: basic slug (/ → _, : → _).
    """
    lower = model_id.lower()
    # 1) Exact match first — short aliases bind here
    if lower in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[lower]
    # 2) Substring fallback — long-form slugs bind here
    for pattern, display in MODEL_DISPLAY_NAMES.items():
        if pattern in lower:
            return display
    # 3) Slug fallback
    return model_id.replace("/", "_").replace(":", "_")
