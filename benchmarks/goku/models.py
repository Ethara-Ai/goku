"""Pydantic models for the Goku evaluation benchmark."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RubricType = Literal[
    "probe_file_exists",
    "probe_file_contains",
    "probe_dir_exists",
    "shell_succeeds_real",
    "response_contains",
    "response_regex_present",
    "response_criteria",
    "response_not_criteria",
]

RubricCategory = Literal[
    "CORRECTNESS",
    "FORMAT",
    "BEHAVIOR",
    "MM_REASONING",
    "HALLUCINATION",
    "STYLE",
]

# Task input media category. Each task is siloed: a `pdf` task may only ship
# PDFs, `image` only images, `video` only videos. The loader enforces this
# at task-discovery time so violations surface early. `mixed` is allowed for
# legacy tasks shipped before the category split — new tasks should pick one.
TaskCategory = Literal["pdf", "image", "video", "mixed"]


class RubricItem(BaseModel):
    """A single rubric item from rubrics.jsonl."""

    number: int = Field(..., ge=1, description="1-based rubric item number")
    type: RubricType
    category: RubricCategory
    points: int = Field(..., description="+5, +3, -5, or -3")
    importance: Literal["mandatory", "nice_to_have"]
    criterion: str = Field(..., min_length=1)

    # Type-specific optional fields
    paths: list[str] | None = None  # probe_file_exists, probe_dir_exists
    path: str | None = None  # probe_file_contains
    pattern: str | None = None  # probe_file_contains, response_regex_present
    ignore_case: bool = False  # probe_file_contains
    raw_shell: str | None = None  # shell_succeeds_real
    needles: list[str] | None = None  # response_contains
    source: dict[str, Any] | list[dict[str, Any]] | None = None  # factuality


class JudgeVerdict(BaseModel):
    """Per-judge verdict for a single rubric item (council mode only)."""

    judge_model: str
    passed: bool
    judge_rationale: str
    judge_cost_usd: float = Field(default=0.0, ge=0.0)
    # Populated when a judge call fails (timeout, API error, etc.) — its
    # vote is treated as a conservative `passed=False` in aggregation, but
    # the reason is preserved so split verdicts are debuggable.
    error: str | None = None


class ScorerResult(BaseModel):
    """Result of scoring a single rubric item."""

    number: int = Field(..., ge=1)
    passed: bool
    judge_rationale: str
    points_awarded: int
    # USD cost of the LLM judge call for this item (0 for deterministic types
    # like probe_* / shell_* / response_contains / response_regex_present).
    # Populated by LiteLLM's `response_cost` for response_criteria /
    # response_not_criteria items. In council mode this is the SUM across
    # all judges that contributed.
    judge_cost_usd: float = Field(default=0.0, ge=0.0)

    # Council-mode fields (None for single-judge runs — preserves backward
    # compatibility with the existing scores.jsonl schema). Populated only
    # when the rubric was scored via score_llm_judge_council().
    per_judge_verdicts: list[JudgeVerdict] | None = None
    vote: str | None = Field(
        default=None,
        description="Council pass vote ratio, e.g. '2/3'. None in single-judge mode.",
    )
    disagreement: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Council split count: 0 if all judges agreed, 1+ for any split. "
            "None in single-judge mode."
        ),
    )
    consensus: Literal["unanimous", "majority"] | None = Field(
        default=None,
        description=(
            "Council consensus type: 'unanimous' iff all judges agreed; "
            "'majority' iff 2/3 (or N/N-1) agreed. None in single-judge mode."
        ),
    )


class TaskScore(BaseModel):
    """Aggregated score for a single task run."""

    awarded: int
    max_total: int
    raw_score: float
    per_task_score: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    items: list[ScorerResult]
    # Sum of judge_cost_usd across all items in `items`. Only populated for
    # tasks where at least one rubric item is LLM-judged.
    judge_cost_usd: float = Field(default=0.0, ge=0.0)


class GokuEvalInstance(BaseModel):
    """A Goku task instance loaded from a task folder."""

    id: str
    instruction: str
    rubric_items: list[RubricItem]
    input_files: list[str] = Field(
        default_factory=list, description="Absolute paths to input media files"
    )
    # Set by the loader from `task_category` in rubrics.jsonl header (or
    # auto-inferred from input file extensions if absent). Drives per-category
    # validation: pdf tasks may not ship video, etc.
    task_category: TaskCategory = Field(
        default="mixed",
        description="Input media category: pdf | image | video | mixed (legacy).",
    )


class BenchmarkReport(BaseModel):
    """Benchmark-level aggregation across all tasks for a single model."""

    model_id: str
    mean_per_task_score: float = Field(
        ..., ge=0.0, le=1.0, description="Primary headline metric"
    )
    mean_raw_score: float = Field(..., description="Can be negative")
    pass_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Mean per-run pass rate across tasks"
    )
    pass_at_3: float = Field(
        ..., ge=0.0, le=1.0, description="Unbiased estimator (lenient)"
    )
    pass_hat_3: float = Field(
        ..., ge=0.0, le=1.0, description="Strict: all 3 runs must pass"
    )
    total_tasks: int = Field(..., ge=0)
    # Cost + token aggregates, summed across every (task, run) for this model.
    # Sourced from each output.jsonl line's `metrics` field (populated by
    # OpenHands/LiteLLM during inference). Default 0 if metrics are absent.
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    total_prompt_tokens: int = Field(default=0, ge=0)
    total_completion_tokens: int = Field(default=0, ge=0)
    total_cache_read_tokens: int = Field(default=0, ge=0)
    total_cache_write_tokens: int = Field(default=0, ge=0)
    mean_cost_per_run_usd: float = Field(default=0.0, ge=0.0)
    total_runs_with_metrics: int = Field(
        default=0, ge=0,
        description="How many (task, run) outputs contributed cost data — "
                    "useful to spot missing metrics when this is < total_tasks * n_runs."
    )
    # Judge LLM cost, summed across every (task, run) for this model.
    # 0 for older scores.jsonl files (judge cost only tracked from this
    # version onward) — re-run the batch to populate.
    total_judge_cost_usd: float = Field(default=0.0, ge=0.0)
    # Per-category mean per_task_score, computed by weighting each rubric
    # item's outcome by its category. Used to evaluate Tab-3 difficulty
    # conformance (non-FORMAT average ≤ 0.7 for at least one agent).
    mean_score_by_category: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-RubricCategory mean of (awarded / max for items in that "
            "category). FORMAT vs non-FORMAT split surfaces Tab-3 difficulty."
        ),
    )
    mean_non_format_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Mean per_task_score across all non-FORMAT rubric items.",
    )
    tab3_difficulty_target_hit: bool = Field(
        default=False,
        description=(
            "True iff this model's mean_non_format_score is ≤ 0.7 "
            "(Tab-3 difficulty target). Compare across reports for the "
            "{gpt-5.5, claude-opus-4.7, gemini-3.1-pro} cohort: at least "
            "one must hit True for the dataset to qualify as on-target."
        ),
    )
