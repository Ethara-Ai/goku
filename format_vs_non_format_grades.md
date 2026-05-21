# Format vs. Non-FORMAT Grade Comparison

**Delivery:** `delivery/MM Agentic Pilot Samples-2026-05-19/`
**Tasks:** `task_ae0bc0ef3ca7ff1d`, `task_ff5c9742c645e2cf`, `task_da7c7053f0e1ff1d`

## Table 1 — Per-run scores (every task × model × run)

| Task | Model | Run | All rubrics | No FORMAT | Δ |
|---|---|---|---:|---:|---:|
| task_ae0bc0ef3ca7ff1d | claude-opus | run_1 | 1.0000 | 1.0000 | +0.0000 |
| task_ae0bc0ef3ca7ff1d | claude-opus | run_2 | 1.0000 | 1.0000 | +0.0000 |
| task_ae0bc0ef3ca7ff1d | gemini-3.1 | run_1 | 0.6389 | 0.5000 | −0.1389 |
| task_ae0bc0ef3ca7ff1d | gemini-3.1 | run_2 | 0.9167 | 0.8846 | −0.0321 |
| task_ae0bc0ef3ca7ff1d | gpt5.5 | run_1 | 1.0000 | 1.0000 | +0.0000 |
| task_ae0bc0ef3ca7ff1d | gpt5.5 | run_2 | 1.0000 | 1.0000 | +0.0000 |
| task_ff5c9742c645e2cf | claude-opus | run_1 | 0.8000 | 0.7143 | −0.0857 |
| task_ff5c9742c645e2cf | claude-opus | run_2 | 0.8000 | 0.7143 | −0.0857 |
| task_ff5c9742c645e2cf | gemini-3.1 | run_1 | 0.8000 | 0.7143 | −0.0857 |
| task_ff5c9742c645e2cf | gemini-3.1 | run_2 | 0.8000 | 0.7143 | −0.0857 |
| task_ff5c9742c645e2cf | gpt5.5 | run_1 | 0.7000 | 0.5714 | −0.1286 |
| task_ff5c9742c645e2cf | gpt5.5 | run_2 | 0.9000 | 0.8571 | −0.0429 |
| task_da7c7053f0e1ff1d | claude-opus | run_1 | 1.0000 | 1.0000 | +0.0000 |
| task_da7c7053f0e1ff1d | claude-opus | run_2 | 0.9091 | 0.9000 | −0.0091 |
| task_da7c7053f0e1ff1d | gemini-3.1 | run_1 | 0.6364 | 0.6000 | −0.0364 |
| task_da7c7053f0e1ff1d | gemini-3.1 | run_2 | 0.8182 | 0.8000 | −0.0182 |
| task_da7c7053f0e1ff1d | gpt5.5 | run_1 | 0.6364 | 0.6000 | −0.0364 |
| task_da7c7053f0e1ff1d | gpt5.5 | run_2 | 0.8182 | 0.8000 | −0.0182 |

## Table 2 — Least-performing model per task (average of both runs)

| Task | Least-performing model | All rubrics (2-run avg) | No FORMAT (2-run avg) |
|---|---|---:|---:|
| task_ae0bc0ef3ca7ff1d | gemini-3.1 | 0.7778 | 0.6923 |
| task_ff5c9742c645e2cf | claude-opus / gemini-3.1 / gpt5.5 (tied) | 0.8000 | 0.7143 |
| task_da7c7053f0e1ff1d | gemini-3.1 / gpt5.5 (tied) | 0.7273 | 0.7000 |
