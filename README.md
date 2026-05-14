# Goku — Multimodal Agentic Evaluation Benchmark

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Goku is a benchmark harness for evaluating AI agents on multi-step, multimodal
tasks. An agent receives a natural-language prompt plus one or more attached
images/PDFs/videos, runs inside a sandboxed Docker workspace, produces files
and a text response, and is graded against a per-task rubric (deterministic
file/shell checks + LLM-judged criteria).

---

## What you get

- **`goku-infer`** — run agents on tasks, score them, write per-task results
- **`goku-eval`** — aggregate scores across models and runs, export a clean
  delivery folder
- **`run_batch.sh`** — overnight batch runner: many tasks × many models ×
  many runs, with parallelism, Docker cleanup, resume support, and automatic
  delivery export

---

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/) ≥ 0.8.13** — Python package manager
- **Docker** with the daemon running (the agent runs inside a sandbox container)
- API credentials for whatever models you want to evaluate:
  - Anthropic / AWS Bedrock (for Claude)
  - OpenAI (for GPT)
  - Google AI (for Gemini)
  - A judge model (e.g. Kimi via Bedrock — used to score `response_criteria`
    rubric items)

---

## Install

```bash
git clone <this-repo-url> goku
cd goku
make build          # syncs submodules, runs `uv sync --dev`, sets up pre-commit
make test           # runs unit tests; should report 97 passed
```

---

## Configure LLMs

Drop one JSON config per model into `.llm_config/`. See
[`.llm_config/example.json`](.llm_config/example.json) for the schema.

```json
{
  "model": "openai/gpt-5.5",
  "base_url": "https://api.openai.com/v1",
  "api_key": "sk-..."
}
```

For Bedrock models, `model` is the full ARN-prefixed identifier:
`bedrock/converse/arn:aws:bedrock:<region>:...:application-inference-profile/<id>`

Because that ARN is ugly in directory names, you can add an optional
`display_name` field to override how the model appears in `eval_outputs/`,
`delivery/`, and reports:

```json
{
  "model": "bedrock/converse/arn:aws:bedrock:...:application-inference-profile/abc123",
  "display_name": "claude-opus-4.7",
  "api_key": "..."
}
```

If omitted, the LLM config's **filename stem** is used (e.g.
`.llm_config/claude-opus-4.7.json` → `claude-opus-4.7`). Falls back to
`model.replace("/", "_")` only if neither is available.

`run_batch.sh` **auto-discovers** configs in `.llm_config/` and classifies them
by the `model` field:

| Filename pattern OR model contains | Role |
|---|---|
| `claude` / `opus` / `sonnet` | OPUS_CONFIG |
| `gpt` / `openai` | GPT_CONFIG |
| `gemini` | GEMINI_CONFIG |
| `kimi` / `moonshot` / filename has `judge` | JUDGE_CONFIG |

`example.json` and `template.json` are skipped during discovery.

You can override any role explicitly:
```bash
export OPUS_CONFIG=/path/to/my-opus.json
```

---

## Author a dataset

Each task lives in its own folder named `task_<unique_id>/` (we use
`task_<sha256[:16]>` for stability; any unique string works). The folder must
contain:

```
dataset/task_<id>/
├── instruction.md          # natural-language user prompt
├── rubrics.jsonl           # one JSON rubric item per line
└── data/
    └── input_files/        # ALL media given to the agent at runtime
```

See **[`sample_tasks/task_template/`](sample_tasks/task_template/)** for a
working example with annotated rubrics. Copy that folder to start a new task.

Two **hard rules** from the spec:

1. `instruction.md` must use **bare filenames only** — no `results/`,
   `/workspace/`, or `/home/` prefixes. Write it as a natural human message.
2. The agent's output filename in the rubric (`paths: ["foo.json"]`) is only
   valid if `instruction.md` specifies that exact filename. If the prompt is
   open-ended ("save it as a file"), use LLM-judged `response_criteria`
   instead.

The loader **fails loudly** if it can't parse a task — no silent skips.

---

## Run

### Option 1: batch runner (recommended)

Evaluates every task in `dataset/` across every model you have a config for,
3 runs per model per task.

```bash
bash run_batch.sh                                 # full batch
bash run_batch.sh --tasks task_abc,task_def       # specific tasks
bash run_batch.sh --models opus,gemini            # subset of models
bash run_batch.sh --runs 1 --parallel 1           # quick single-run, sequential
bash run_batch.sh --dry-run                       # preview without executing
bash run_batch.sh --help                          # all options
```

The script handles Docker cleanup, run timeouts (default 20 min/run), retries
(2 attempts/run), and resume (skips already-completed `(task, model, run)`
combinations).

### Option 2: single-shot CLI

```bash
uv run goku-infer .llm_config/your-model.json \
    --tasks-dir dataset \
    --task task_abc \
    --runs 1 \
    --workspace docker \
    --max-iterations 30 \
    --num-workers 1 \
    --output-dir eval_outputs/run_1 \
    --critic pass \
    --judge-llm-config .llm_config/your-judge.json
```

After all runs finish, generate the delivery export:

```bash
uv run goku-eval \
    --output-dir eval_outputs \
    --models bedrock_converse_arn gemini openai \
    --runs 3 \
    --tasks-dir dataset \
    --export-delivery delivery
```

`--models` here are substrings of the inference output directory names (e.g.
`bedrock_converse_arn` matches the ARN-named opus dir). `goku-eval` maps these
to clean folder names in delivery (`claude-opus`, `gemini-3.1`, `gpt5.5`).

### Re-scoring after editing rubrics (no inference re-run)

If you edit `dataset/<task>/rubrics.jsonl` after a batch has executed, you can
re-score the existing agent outputs against the new rubrics **without paying
for agent inference again** — only the judge LLM is invoked (for
`response_criteria` / `response_not_criteria` items).

```bash
# Re-score everything
uv run goku-rescore \
    --output-dir eval_outputs \
    --tasks-dir dataset \
    --judge-llm-config .llm_config/kimi-k2.5-judge.json \
    --backup \
    --export-delivery delivery

# Re-score a subset
uv run goku-rescore \
    --output-dir eval_outputs --tasks-dir dataset \
    --tasks task_abc,task_def \
    --models bedrock_converse_arn

# Cheap dry-run: re-score deterministic items only, no judge calls
uv run goku-rescore \
    --output-dir eval_outputs --tasks-dir dataset \
    --skip-llm-judge

# Just list what would be rescored
uv run goku-rescore --output-dir eval_outputs --tasks-dir dataset --dry-run
```

`--backup` saves the original `scores.jsonl` to `scores.before-rescore.jsonl`
so you can compare. `--export-delivery` re-runs the delivery export
afterwards so the packaged scores reflect the new rubrics.

---

## Output layout

```
eval_outputs/                              # raw evaluation outputs
└── run_N/                                      # one per `--runs N` invocation
    └── run_1/
        └── goku/
            └── <model_dir>/                    # named after the model identifier
                ├── output.jsonl                # one line per task (full payload)
                ├── output.critic_attempt_1.jsonl
                ├── conversations/              # tarred conversation logs
                ├── logs/                       # per-instance log files
                └── task_<id>/
                    ├── scores.jsonl
                    └── results/                # files the agent saved
                        ├── bash_events/        # full bash trace
                        └── <agent-saved files>

delivery/
└── MM Agentic Pilot Samples-<YYYY-MM-DD>/
    └── tasks/
        └── task_<id>/
            ├── instruction.md                  # copied from dataset/
            ├── rubrics.jsonl                   # copied from dataset/
            ├── data/input_files/               # copied from dataset/
            └── runs/
                └── <clean_model_name>/         # claude-opus, gemini-3.1, gpt5.5
                    └── run_N/
                        ├── output.jsonl
                        ├── scores.jsonl
                        └── results/
                            ├── bash_events/
                            └── <agent-saved files>
```

---

## Scoring

Each rubric item is graded independently. Item types:

| Type | Graded by | Notes |
|---|---|---|
| `probe_file_exists` | filesystem | `paths: ["file.json"]` — searches recursively under the agent's output dir |
| `probe_file_contains` | regex | `path` + `pattern`, with optional `ignore_case` |
| `probe_dir_exists` | filesystem | `paths: ["images"]` |
| `shell_succeeds_real` | exit code | `raw_shell` bash one-liner, 30s timeout, cwd = output dir |
| `response_contains` | substring | `needles: ["..."]` |
| `response_regex_present` | regex | `pattern` matched against agent's text response |
| `response_criteria` | LLM judge | natural-language assertion graded 0/1 |
| `response_not_criteria` | LLM judge | hallucination detector: criterion matches = penalty triggers |

Per-task aggregation:

```
awarded   = sum(passed positive points) − sum(triggered negative |points|)
max_total = sum(positive points)
raw_score = awarded / max_total                  # can be negative
per_task_score = clip(raw_score, 0.0, 1.0)
pass = all mandatory positives passed AND no mandatory negative triggered
```

Benchmark-level (across tasks):
- `mean_per_task_score` — primary headline metric
- `pass_rate`, `pass@3` (Codex estimator), `pass^3` (strict: all 3 runs pass)

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `No LLM config discovered for model alias: 'opus'` | `.llm_config/` doesn't contain a JSON whose `model` field matches the role. Either add one or `export OPUS_CONFIG=/path/to/your.json` |
| `Docker daemon not running` | Start Docker Desktop / `systemctl start docker` |
| `Task X: instruction.md contains forbidden path pattern '/workspace/'` | Rewrite the prompt to use bare filenames only |
| `discover_tasks: N task(s) failed to load` | The loader will name the offending task and the parse error. Fix the task's `rubrics.jsonl` or `instruction.md` |
| Bedrock judge returns `region` errors | Set `AWS_REGION_NAME` env var, or add `"aws_region_name": "..."` to your judge LLM config, or use an ARN-prefixed model (region is auto-parsed) |
| Agent receives no images | Check that media is in `data/input_files/` (not `data/`). The loader only looks in `data/input_files/` |
| Delivery folder has nested `results/results/` | Your prompt mentions `results/` — drop that prefix and use a bare filename, then re-export |

---

## Project structure

```
goku/
├── benchmarks/goku/                # the benchmark module
│   ├── run_infer.py                # inference pipeline
│   ├── eval_infer.py               # report + delivery export
│   ├── scoring.py                  # per-task score aggregation
│   ├── task_loader.py              # dataset discovery + validation
│   ├── models.py                   # pydantic schemas
│   ├── config.py                   # model display-name lookup
│   ├── scorers/
│   │   ├── deterministic.py        # probe_*, shell_*, response_contains/regex
│   │   └── llm_judge.py            # response_criteria, response_not_criteria
│   └── tests/                      # 97 unit tests
├── dataset/                        # YOUR tasks go here
├── sample_tasks/task_template/     # working example to copy
├── .llm_config/                    # YOUR LLM config JSONs
├── eval_outputs/              # raw inference outputs (created on run)
├── delivery/                       # packaged delivery folder (created on export)
├── run_batch.sh                    # batch runner
├── Makefile                        # build, format, lint, test, clean
└── pyproject.toml
```

---

## License

MIT — see [LICENSE](LICENSE).
