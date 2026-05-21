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
- **`ffmpeg`** on `PATH` — required for tasks with video inputs. The harness
  extracts evenly-spaced keyframes via `ffmpeg` so the agent and judge see
  the video as a sequence of images. macOS: `brew install ffmpeg`; Debian/Ubuntu:
  `apt install ffmpeg`. Skip if you don't author video-input tasks.
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
make test           # runs unit tests; should report 161 passed (goku module)
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
  "model_canonical_name": "anthropic.claude-opus-4-7",
  "api_key": "..."
}
```

If omitted, the LLM config's **filename stem** is used (e.g.
`.llm_config/claude-opus-4.7.json` → `claude-opus-4.7`). Falls back to
`model.replace("/", "_")` only if neither is available.

**`model_canonical_name`** (Bedrock-only, optional but recommended for
multimodal tasks): the Bedrock application-inference-profile ARN is opaque —
nothing in it identifies the underlying model family. The multimodal router
consults `model_canonical_name` to pick the right per-provider PDF block
shape (`anthropic.*` → `document` block; `openai.*` / `gemini.*` → `file`
block; everything else → render PDF pages to images). If omitted, the
router falls back to rendering — slightly lossier on dense text, but safe.

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

## Multimodal inputs (images, PDFs, videos)

Any file dropped into a task's `data/input_files/` is uploaded to the agent's
workspace **and** attached to the agent's initial multimodal turn. The agent
*and* the judge both see the file — the same one — so visual rubrics
(`response_not_criteria` hallucination checks especially) can be graded
against ground truth, not against the agent's own description.

Same multimodal pipeline applies to the agent's **OUTPUT** files: if the
agent produces a PDF, image, or video into `/workspace/results/`, those
files are attached to the judge under a separate `=== OUTPUT MEDIA ===`
section block. Without this, output rubrics (e.g. "the agent's saved PDF
contains a Conclusion section") could only be graded against the agent's
text description of its own output — the judge would have to bluff.

### Per-file routing

| File type | Agent (Claude / GPT / Gemini) | Judge (Kimi by default) |
|---|---|---|
| **Image** (PNG/JPG/JPEG/GIF/WEBP) | Native `image_url` block | Native `image_url` block |
| **PDF** | Native per provider: `document` block (Claude) / `file` block (GPT, Gemini) | If judge supports native PDF: native block. If not (Kimi-Bedrock): **rendered to page PNGs via `pypdfium2`** at 200 DPI and attached as images. |
| **Video** (MP4/MOV/WEBM/AVI/MKV) | **Uniformly extracted to 60 keyframes via `ffmpeg`**, attached as images. Symmetric across all 3 agent models so video task scores stay comparable. | Same: 60 keyframes via `ffmpeg`. |
| Other (CSV, JSON, MD, code, …) | Uploaded to `/workspace/` only — agent reads via shell/tool calls. | Not attached; judge sees the agent's text summary only. |

### Task categories — strict siloing

Tasks fall into one of three categories, declared via a `task_category` header
line in `rubrics.jsonl` (the loader treats lines without a `number` field as
header records):

```jsonl
{"task_category": "pdf"}
{"number": 1, "type": "probe_file_exists", ...}
{"number": 2, "type": "response_criteria", ...}
```

Allowed values: **`pdf`** | **`image`** | **`video`** | **`mixed`** (legacy
only). A `pdf` task may only ship `.pdf` files in `data/input_files/`; an
`image` task only image extensions; a `video` task only video extensions.
Mixing categories within a single task is rejected at load time so the
benchmarking comparison stays clean. If no header is present, the loader
infers the category from extensions and downgrades violations to warnings
(grandfathered legacy behavior).

### Hard limits — per category (enforced at task load)

| Category | Per-file | Files per task | Total payload |
|---|---|---|---|
| **PDF** | 100 pages, 30 MB | 1–3 PDFs | ≤ 40 MB |
| **Image** | 5 MB per file, ≤ 4096×4096 px | up to 20 images | ≤ 40 MB |
| **Video** | 60 min, 200 MB, 1080p | 1 video | ≤ 200 MB; 60 keyframes auto-extracted |

These caps live in `benchmarks/goku/media_render.py` (`MAX_PDF_BYTES`,
`MAX_VIDEO_BYTES`, `MAX_IMAGE_BYTES`, `MAX_VIDEO_DURATION_SEC`). The loader's
`_validate_input_files_for_category` enforces them when a category is
explicitly declared and warns when inferred.

### Judge prompt — input vs output separation

The judge's multimodal payload is built as a labeled content array:

```
[ text(prompt + instructions), 
  text("=== INPUT MEDIA (the task fixture given to the agent) ==="),
    image/document/keyframe blocks for the INPUT,
  text("=== OUTPUT MEDIA (files the agent produced as its work product) ==="),
    image/document/keyframe blocks for the OUTPUT ]
```

The prompt's instructions explicitly tell the judge to "look at INPUT media
for grounding what was in the task, OUTPUT media for what the agent
produced, and compare them side-by-side if the criterion spans both." A
dedicated test suite (`TestInputOutputMediaSeparation` in
`tests/test_llm_judge.py`) pins this invariant, and an empirical probe
verified the Kimi judge cites the correct section in its rationale for
output-correctness, input-grounding, and cross-media hallucination rubrics.

### Bedrock opaque-ARN gotcha

Bedrock application-inference-profile ARNs contain no provider markers, so
add a `model_canonical_name` field to the LLM config
(e.g. `"anthropic.claude-opus-4-7"`) so the multimodal router can pick the
right per-provider block shape. See [`Configure LLMs`](#configure-llms).

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
bash run_batch.sh --tasks task_abc --rerun        # force re-inference (see below)
bash run_batch.sh --help                          # all options
```

The script handles Docker cleanup, run timeouts (default 20 min/run), retries
(2 attempts/run), and resume (skips already-completed `(task, model, run)`
combinations).

**Forcing re-inference after editing a task.** Resume is keyed on the
harness's `output.jsonl` / `output.critic_attempt_*.jsonl`, so editing a
prompt or rubric and re-running silently skips the task with
`"No instances to process"`. Pass `--rerun` together with `--tasks` to strip
those resume-state entries (and archive the previous per-task outputs) before
launching — required after any prompt or rubric edit. `--rerun` refuses to
run without `--tasks` to avoid silently wiping the whole batch.

The script exits with **`2`** when zero tasks completed (e.g. silent harness
skip, Modal 500s, credential rejection) so a watcher can distinguish that
from a normal partial-success exit `1` or full-success exit `0`.

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

### Backfilling `response.md` on an existing delivery

`goku-eval --export-delivery` writes `results/response.md` for every run as
part of the export. If you already have a delivery folder from before this
landed (or you re-ran inference and want to refresh `response.md` without
re-exporting the whole delivery), use the standalone backfill CLI:

```bash
# Backfill every missing response.md, pulling from eval_outputs/
uv run goku-write-response-md delivery/MM\ Agentic\ Pilot\ Samples-2026-05-15/ \
    --eval-outputs-root eval_outputs

# Force-overwrite existing response.md (e.g. after re-inference)
uv run goku-write-response-md delivery/MM\ Agentic\ Pilot\ Samples-2026-05-15/ \
    --eval-outputs-root eval_outputs --force

# Limit to specific tasks / models, or preview without writing
uv run goku-write-response-md delivery/.../ --eval-outputs-root eval_outputs \
    --tasks task_abc,task_def --models claude-opus --dry-run
```

The script is idempotent — repeated runs against an unchanged trajectory
produce the same `response.md`. Source trajectories must live in
`eval_outputs/` (delivery never contains them).

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
                        ├── scores.jsonl
                        └── results/
                            ├── response.md     # model's final natural-language response
                            └── <agent-saved files>
```

The delivery folder is **intentionally narrower** than `eval_outputs/`:
`output.jsonl` (the full OpenHands trajectory) and `results/bash_events/`
(raw tool-call log) stay in `eval_outputs/` for debugging and are not
shipped — per the doc spec the deliverable is scores + the model's response
+ agent-produced artifacts.

`response.md` is extracted from the source `output.jsonl` (the agent's
`FinishAction` message, with a fallback to the last agent text event for
abnormally-terminated runs). It carries the agent's own markdown verbatim
with no scaffolding.

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
| `No instances to process` and the batch finishes in seconds | Harness resume-state is still holding the task as "done". Re-run `run_batch.sh --tasks <keys> --rerun` to strip `output.jsonl` + `output.critic_attempt_*.jsonl` entries before launching |
| `BATCH FAILED — 0 of N tasks completed` (exit 2) | Either silent resume-state skip (use `--rerun`), Docker/Modal connectivity issues, or LLM-config credential rejection. Per-model logs are in `batch_logs/<batch_id>/` |
| Delivery folder has no `response.md` in `results/` | Pre-`response.md`-feature delivery — backfill with `uv run goku-write-response-md <delivery_root> --eval-outputs-root eval_outputs` (see Re-scoring section) |
| `ffmpeg not found on PATH` | Install ffmpeg (`brew install ffmpeg` on macOS, `apt install ffmpeg` on Debian). Required only for video-input tasks. |
| `ModuleNotFoundError: pypdfium2` | Run `uv sync` to pull the dep (added for the multimodal-judge PDF fallback path). |
| Judge rationale contains `[JUDGE MEDIA WARNINGS — judge did NOT see ...]` | A media file couldn't be attached to the judge (oversized, corrupt video, missing renderer). The note names the file — fix the source or raise the cap in `benchmarks/goku/scorers/llm_judge.py`. |
| `This model doesn't support documents.` (Bedrock) | Judge model doesn't accept native PDFs (Kimi via Bedrock). Set `model_canonical_name` on the LLM config so the multimodal router selects the pypdfium2 fallback path instead of the native `document` block. |
| Claude Opus 4.7 returns `temperature is deprecated` | Opus 4.7 rejects any non-default `temperature` / `top_p` / `top_k`. If using Claude as judge, omit the `"temperature"` field from the LLM config. |
| `Task X: input_files violate per-category limits` | A task declares `task_category` in `rubrics.jsonl` but its files violate the cap (wrong extension, oversized, video too long). The error message names the file and the cap. Either fix the file or change/remove the `task_category` header. |
| `Task X: invalid task_category 'foo'` | `task_category` must be one of `pdf` / `image` / `video` / `mixed`. Fix the header line. |
| Legacy task loads with a WARNING about cap violations | Task has no `task_category` header, the loader inferred one from extensions, and a file exceeds the inferred-category cap. Either compress/downscale the file or add an explicit `task_category` header so the loader uses strict validation. |

---

## Project structure

```
goku/
├── benchmarks/goku/                # the benchmark module
│   ├── run_infer.py                # inference pipeline (agent path)
│   ├── eval_infer.py               # report + delivery export
│   ├── rescore.py                  # re-judge existing trajectories (no agent re-run)
│   ├── scoring.py                  # per-task score aggregation
│   ├── task_loader.py              # dataset discovery + validation
│   ├── models.py                   # pydantic schemas
│   ├── config.py                   # model display-name lookup
│   ├── media_render.py             # PDF→images (pypdfium2), video→keyframes (ffmpeg)
│   ├── media_adapters.py           # per-provider PDF block routing
│   ├── response_extractor.py       # final-response extraction for delivery
│   ├── write_response_md.py        # backfill response.md on existing deliveries
│   ├── scorers/
│   │   ├── deterministic.py        # probe_*, shell_*, response_contains/regex
│   │   └── llm_judge.py            # response_criteria, response_not_criteria
│   └── tests/                      # 161 unit tests
├── benchmarks/utils/
│   ├── sdk_patches.py              # runtime extension of OpenHands SDK Message
│   │                               #   for native PDF content blocks
│   ├── httpx_patches.py            # follow_redirects=True for SDK RemoteWorkspace
│   └── sitecustomize.py            # applies sdk_patches + httpx_patches at startup
├── vendor/software-agent-sdk/      # OpenHands SDK as a git submodule (untouched)
├── dataset/                        # YOUR tasks go here
├── sample_tasks/task_template/     # working example to copy
├── .llm_config/                    # YOUR LLM config JSONs
├── eval_outputs/                   # raw inference outputs (created on run)
├── delivery/                       # packaged delivery folder (created on export)
├── run_batch.sh                    # batch runner
├── Makefile                        # build, format, lint, test, clean
└── pyproject.toml
```

---

## License

MIT — see [LICENSE](LICENSE).
