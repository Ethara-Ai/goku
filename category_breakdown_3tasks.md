# Category-Level Breakdown — 3 Tasks (per-model + 3-model average)

Per-run FORMAT vs. non-FORMAT score breakdown, per agent model and averaged.
Each score = `clip(awarded / max_total, 0, 1)` over the subset of rubrics
in that category (FORMAT only / non-FORMAT only).

| Task ID | Run | FORMAT rubrics (count + item #s) | Non-FORMAT rubrics (count + item #s) | claude-opus FMT | claude-opus Non-FMT | gemini-3.1 FMT | gemini-3.1 Non-FMT | gpt5.5 FMT | gpt5.5 Non-FMT | 3-model avg FMT | 3-model avg Non-FMT | 3-model avg overall |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| task_ae0bc0ef3ca7ff1d (RSU vesting) | run_1 | 2 (R1, R2) | 7 (R3, R4, R5, R6, R7, R8, R9) | 1.0000 | 1.0000 | 1.0000 | **0.5000** | 1.0000 | 1.0000 | 1.0000 | 0.8333 | 0.8796 |
| task_ae0bc0ef3ca7ff1d (RSU vesting) | run_2 | 2 (R1, R2) | 7 (R3, R4, R5, R6, R7, R8, R9) | 1.0000 | 1.0000 | 1.0000 | 0.8846 | 1.0000 | 1.0000 | 1.0000 | 0.9615 | 0.9722 |
| task_ff5c9742c645e2cf (breakfast) | run_1 | 3 (R1, R2, R3) | 9 (R4, R5, R6, R7, R8, R9, R10, R11, R12) | 1.0000 | 0.7143 | 1.0000 | 0.7143 | 1.0000 | **0.5714** | 1.0000 | 0.6667 | 0.7667 |
| task_ff5c9742c645e2cf (breakfast) | run_2 | 3 (R1, R2, R3) | 9 (R4, R5, R6, R7, R8, R9, R10, R11, R12) | 1.0000 | 0.7143 | 1.0000 | 0.7143 | 1.0000 | 0.8571 | 1.0000 | 0.7619 | 0.8333 |
| task_da7c7053f0e1ff1d (notification system) | run_1 | 1 (R1) | 11 (R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12) | 1.0000 | 1.0000 | 1.0000 | **0.6000** | 1.0000 | **0.6000** | 1.0000 | 0.7333 | 0.7576 |
| task_da7c7053f0e1ff1d (notification system) | run_2 | 1 (R1) | 11 (R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12) | 1.0000 | 0.9000 | 1.0000 | 0.8000 | 1.0000 | 0.8000 | 1.0000 | 0.8333 | 0.8485 |

Bolded non-FMT cells flag scores ≤ 0.7 — useful when checking whether the
Tab-3 difficulty target ("non-FORMAT avg ≤ 0.7 for at least one agent") is
met on that run.

## Tab-3 target check at a glance

| Task | Lowest non-FORMAT score across all (model, run) | Tab-3 satisfied (≤ 0.7)? |
|---|---:|:---:|
| task_ae0bc0ef3ca7ff1d (RSU vesting) | **0.5000** (gemini-3.1, run_1) | ✅ |
| task_ff5c9742c645e2cf (breakfast) | **0.5714** (gpt5.5, run_1) | ✅ |
| task_da7c7053f0e1ff1d (notification system) | **0.6000** (gemini-3.1 & gpt5.5, run_1) | ✅ |
