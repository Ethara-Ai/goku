#!/usr/bin/env bash
#
# Goku Benchmark — Overnight Batch Runner
#
# Runs all tasks in dataset/ across 3 models × 3 runs with:
#   - Parallel execution (1 model per task concurrently by default, configurable)
#   - Docker ghost container cleanup between tasks
#   - Per-run timeout watchdog (kills stuck processes)
#   - Resume support (skips completed task+model+run combos)
#   - Automatic delivery export after completion
#   - Structured logging to file + stdout
#
# Usage:
#   nohup bash run_batch.sh > batch_run.log 2>&1 &
#   nohup bash run_batch.sh --dry-run > batch_dry.log 2>&1 &
#   bash run_batch.sh --tasks task_6c001da3c82d9767,task_c6f4581ec2f2c0dc
#
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Repo root derived from this script's own location — works regardless of
# where the repo is cloned (override with REPO_DIR=... env var if needed).
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATASET_DIR="${DATASET_DIR:-${REPO_DIR}/dataset}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_DIR}/eval_outputs}"
DELIVERY_DIR="${DELIVERY_DIR:-${REPO_DIR}/delivery}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/batch_logs}"
LLM_CONFIG_DIR="${LLM_CONFIG_DIR:-${REPO_DIR}/.llm_config}"

# Model configs — auto-discovered from $LLM_CONFIG_DIR by classifying each
# JSON file's "model" field. Override any of these by exporting them before
# running (e.g. `export OPUS_CONFIG=/path/to/your-opus.json`).
#
# Discovery rule (first match wins per role):
#   model contains "claude"/"opus"/"sonnet"           → OPUS_CONFIG (agent)
#   model contains "gpt"/"openai"                     → GPT_CONFIG  (agent)
#   model contains "gemini-3"                         → GEMINI_CONFIG (agent)
#   model contains "gemini-3.5-flash" / filename
#   contains "flash"/"judge"                          → JUDGE_CONFIG
#
# IMPORTANT: Kimi/Moonshot is NO LONGER a valid judge. Bedrock-Kimi has a
# request-body-size limit (~few MB) that causes hard failures whenever the
# agent produces multi-MB output media — the judge call returns 400 and
# every LLM-judged rubric on that task gets a force-fail verdict. Verified
# in audit. Gemini-3.5-flash is the only supported judge today.
# Kimi configs are explicitly skipped during discovery below.
discover_llm_configs() {
    if [[ ! -d "$LLM_CONFIG_DIR" ]]; then
        return 0
    fi
    local discovered
    discovered=$(python3 - "$LLM_CONFIG_DIR" <<'PYEOF'
import glob, json, os, sys

cfg_dir = sys.argv[1]

# Skip files that are obviously templates/examples — they pollute discovery
# if their `model` field references a placeholder model.
SKIP_BASENAMES = {"example.json", "template.json", "sample.json"}

role_keywords = {
    # Judge: ONLY gemini-3.5-flash. Kimi/Moonshot are explicitly rejected
    # below (Bedrock body-size errors on multi-MB agent outputs cause
    # entire judge calls to 400). The "judge" filename token is kept so
    # operators can name their config file e.g. `gemini-judge.json` and
    # still get picked up automatically. The "flash" token covers the
    # default `gemini-3.5-flash.json` naming.
    "JUDGE_CONFIG":  {"filename": ("flash", "judge"),
                       "model":    ("gemini-3.5-flash", "gemini-flash")},
    "OPUS_CONFIG":   {"filename": ("opus", "claude", "sonnet"),
                       "model":    ("claude", "opus", "sonnet")},
    "GPT_CONFIG":    {"filename": ("gpt", "openai"),
                       "model":    ("gpt", "openai")},
    # Agent gemini: must NOT match the flash judge — explicitly exclude
    # the flash variant from agent discovery so the judge config doesn't
    # accidentally land in the agent slot when both files are present.
    "GEMINI_CONFIG": {"filename": ("gemini-3.1", "gemini-pro", "gemini_3"),
                       "model":    ("gemini-3.1", "gemini-pro")},
}

# Hard-skip list: filenames whose model field marks them as Kimi/Moonshot
# get dropped from discovery entirely. This is the operator-visible
# enforcement of "Kimi is not a valid judge anywhere in the pipeline."
_KIMI_SKIP_MARKERS = ("kimi", "moonshot")

files = []
skipped_kimi = []
for path in sorted(glob.glob(os.path.join(cfg_dir, "*.json"))):
    base = os.path.basename(path).lower()
    if base in SKIP_BASENAMES or base.startswith("_"):
        continue
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        continue
    model_lower = str(data.get("model", "")).lower()
    # Kimi/Moonshot is rejected wholesale — neither valid as judge nor as
    # agent in this pipeline. Body-size limits cause silent score-suppression.
    if any(k in base or k in model_lower for k in _KIMI_SKIP_MARKERS):
        skipped_kimi.append(base)
        continue
    files.append((path, base, model_lower))

if skipped_kimi:
    # Emit a comment so the operator sees it in the eval'd shell context.
    # Doesn't affect bash execution; `eval` ignores comment-only output lines.
    for b in skipped_kimi:
        print(f"# skipped Kimi/Moonshot config: {b} (not a supported judge)")

# Score each (file, role) pair: filename match = 2, content match = 1.
# Pick the highest-scoring file per role, breaking ties by original order.
roles = {}
for role, kw in role_keywords.items():
    best = (-1, None)
    for idx, (path, base, model) in enumerate(files):
        score = 0
        if any(k in base for k in kw["filename"]):
            score += 2
        if any(k in model for k in kw["model"]):
            score += 1
        if score > best[0]:
            best = (score, path)
    if best[0] > 0:
        roles[role] = best[1]

# Each file can only be assigned to one role — prefer the highest-confidence
# role. Resolve conflicts greedily by score.
assigned = {}
for role, path in roles.items():
    if path not in assigned.values():
        assigned[role] = path

for k, v in assigned.items():
    print(f"export {k}={v}")
PYEOF
)
    if [[ -n "$discovered" ]]; then
        eval "$discovered"
    fi
}
discover_llm_configs

# Apply discovered values (env-var overrides win since we used ${VAR:-...} above)
OPUS_CONFIG="${OPUS_CONFIG:-}"
GPT_CONFIG="${GPT_CONFIG:-}"
GEMINI_CONFIG="${GEMINI_CONFIG:-}"
JUDGE_CONFIG="${JUDGE_CONFIG:-}"

# Execution parameters
RUNS_PER_MODEL=3
MAX_ITERATIONS=100         # Bumped from 30 — heavy multimodal tasks (PIL
                           # compositing, multi-file generation) routinely
                           # need 50-80 iterations to complete. 100 leaves
                           # safe headroom under the 60-min conversation cap.
NUM_WORKERS=1              # Workers per model run (keep 1 for Docker stability)
MAX_PARALLEL_MODELS=3      # How many models to run concurrently per task
MAX_PARALLEL_TASKS=1       # How many tasks to run concurrently (1 = sequential)
                           # Total concurrent containers ≈
                           #   MAX_PARALLEL_TASKS × MAX_PARALLEL_MODELS.
                           # Each agent-server container needs ~1–2 GB RAM.
                           # Default 1 (sequential tasks) keeps memory under
                           # Docker Desktop's typical 8 GB allocation. Bump
                           # only if Docker has enough RAM headroom.
RUN_TIMEOUT=3600           # Timeout per single run in seconds (60 min).
                           # Empirically calibrated against the Aditya
                           # Joshi 40-min video task on 2026-05-22:
                           #   - Opus 4.7: 16.9 min end-to-end (inference
                           #     + judge)
                           #   - Gemini 3.1: 17.9 min end-to-end
                           #   - GPT-5.5: ran for 70+ min when allowed —
                           #     not because of LLM latency but because
                           #     the model self-loops on ffmpeg contact
                           #     sheets. 60 min caps that unbounded
                           #     refinement; the model still gets a
                           #     reasonable first report and the wrapper
                           #     proceeds to scoring.
                           # Was 5400s (90 min) — too generous; the only
                           # case it covered was an over-iterating
                           # GPT-5.5, and that case produces a marginally
                           # better answer at huge wall-time cost.
                           # Override with --timeout for genuinely long
                           # tasks.
CONTAINER_STARTUP_WAIT=10  # Seconds to wait after Docker cleanup
DOCKER_IMAGE="ghcr.io/openhands/agent-server:0f70e4e-nikolaik_s_python-nodejs_tag_python3.12-nodejs22-source"

# Retry
MAX_RETRIES_PER_RUN=2      # Retry a failed run this many times

# ─────────────────────────────────────────────────────────────────────────────
# CLI ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

DRY_RUN=false
SPECIFIC_TASKS=""
SPECIFIC_MODELS=""
SKIP_EXPORT=false
RERUN=false  # When true, strip prior resume-state entries for --tasks before launching

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --tasks)
            SPECIFIC_TASKS="$2"
            shift 2
            ;;
        --models)
            SPECIFIC_MODELS="$2"
            shift 2
            ;;
        --runs)
            RUNS_PER_MODEL="$2"
            shift 2
            ;;
        --timeout)
            RUN_TIMEOUT="$2"
            shift 2
            ;;
        --parallel)
            MAX_PARALLEL_MODELS="$2"
            shift 2
            ;;
        --parallel-tasks)
            MAX_PARALLEL_TASKS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --skip-export)
            SKIP_EXPORT=true
            shift
            ;;
        --rerun)
            # Force re-inference of the tasks listed in --tasks by stripping
            # them from harness resume-state files (output.jsonl AND
            # output.critic_attempt_*.jsonl) and archiving any per-task
            # outputs before launching. Required after rubric/prompt edits.
            RERUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: run_batch.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run          Print what would run without executing"
            echo "  --tasks LIST       Comma-separated task keys (default: all in dataset/)"
            echo "  --models LIST      Comma-separated: opus,gpt,gemini (default: all)"
            echo "  --runs N           Runs per model per task (default: 3)"
            echo "  --timeout SEC      Timeout per single run (default: 3600 / 60 min)"
            echo "  --parallel N       Models to run concurrently per task (default: 3)"
            echo "  --parallel-tasks N Tasks to run concurrently (default: 1)"
            echo "                     Total concurrent containers ≈ parallel × parallel-tasks."
            echo "                     Each container uses ~1-2 GB RAM — size to Docker's memory budget."
            echo "  --output-dir DIR   Output directory (default: eval_outputs)"
            echo "  --skip-export      Skip delivery export at end"
            echo "  --rerun            Strip prior resume-state for --tasks before"
            echo "                     launching. Use after editing a task's prompt"
            echo "                     or rubric to force fresh re-inference."
            echo "  --help             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate --rerun usage early — refuse to silently clean everything
if [[ "$RERUN" == "true" && -z "$SPECIFIC_TASKS" ]]; then
    echo "ERROR: --rerun requires --tasks <comma-separated-keys>"
    echo "       (Refusing to silently strip resume-state for every task)"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

cd "$REPO_DIR"

mkdir -p "$OUTPUT_BASE" "$LOG_DIR" "$DELIVERY_DIR"

# Export required env vars
export OPENHANDS_EVAL_AGENT_SERVER_IMAGE="$DOCKER_IMAGE"

# Pull AWS_BEARER_TOKEN_BEDROCK from the judge config (only meaningful when
# the judge is a Bedrock model; harmless otherwise).
if [[ -n "$JUDGE_CONFIG" && -f "$JUDGE_CONFIG" ]]; then
    _judge_key=$(python3 - "$JUDGE_CONFIG" <<'PYEOF' 2>/dev/null || true
import json, sys
try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
    print(d.get("api_key", ""))
except Exception:
    pass
PYEOF
)
    if [[ -n "${_judge_key:-}" ]]; then
        export AWS_BEARER_TOKEN_BEDROCK="$_judge_key"
    fi
    unset _judge_key
fi

# Timestamp for this batch
BATCH_ID=$(date +%Y%m%d_%H%M%S)
BATCH_LOG="${LOG_DIR}/batch_${BATCH_ID}.log"

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

log() {
    local level="$1"
    shift
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [${level}] $*"
    echo "$msg" | tee -a "$BATCH_LOG"
}

log_info()  { log "INFO"  "$@"; }
log_warn()  { log "WARN"  "$@"; }
log_error() { log "ERROR" "$@"; }
log_ok()    { log "OK"    "$@"; }

# ─────────────────────────────────────────────────────────────────────────────
# DOCKER CLEANUP
# ─────────────────────────────────────────────────────────────────────────────

cleanup_docker() {
    # Scoped cleanup: only remove STOPPED containers. Running containers
    # are either (a) ours and still doing legitimate work (e.g. an earlier
    # task's still-running judge phase), or (b) belong to a SIBLING
    # batch_run.sh invocation. Either way, force-killing them mid-flight
    # destroys uncommitted work — the GPT-5.5/Aditya scoring loss on
    # 2026-05-22 happened because a new aman batch's initial cleanup
    # `docker rm -f`'d the prior batch's still-active scoring container.
    #
    # The kill-running-containers safety net is preserved by
    # kill_ghost_containers() further down, which only kills containers
    # that have been running longer than RUN_TIMEOUT.
    #
    # Our containers are named `agent-server-<uuid>` by the OpenHands SDK
    # (vendor/software-agent-sdk/openhands-workspace/.../docker_workspace.py).
    # We match on the `agent-server-` prefix (with the trailing dash) so
    # unrelated containers that merely contain the substring "agent-server"
    # in their name are left alone. We also match by ancestor image as a
    # belt-and-braces measure for the exact image tag we used.
    log_info "Cleaning up stopped agent-server containers..."

    local removed_total=0

    # Statuses we treat as "safe to remove": these are containers that are
    # no longer running, so killing them takes no work away. `running` and
    # `restarting` are deliberately EXCLUDED.
    for status in exited dead created paused; do
        local containers
        containers=$(docker ps -a \
            --filter "status=$status" \
            --filter "ancestor=${DOCKER_IMAGE}" \
            -q 2>/dev/null || true)
        if [[ -n "$containers" ]]; then
            echo "$containers" | xargs docker rm 2>/dev/null || true
            removed_total=$(( removed_total + $(echo "$containers" | wc -l | tr -d ' ') ))
        fi
        # Name-prefix match as fallback for any container whose ancestor
        # image differs from the current ${DOCKER_IMAGE} pin.
        containers=$(docker ps -a \
            --filter "status=$status" \
            --filter "name=^agent-server-" \
            -q 2>/dev/null || true)
        if [[ -n "$containers" ]]; then
            echo "$containers" | xargs docker rm 2>/dev/null || true
            removed_total=$(( removed_total + $(echo "$containers" | wc -l | tr -d ' ') ))
        fi
    done

    if [[ "$removed_total" -gt 0 ]]; then
        log_info "Removed $removed_total stopped agent-server container(s)"
    fi

    # Detect (but do not kill) running containers — useful diagnostic when
    # a sibling batch is active so the operator understands why their slot
    # count looks higher than expected.
    local running
    running=$(docker ps --filter "name=^agent-server-" --format '{{.Names}}' 2>/dev/null || true)
    if [[ -n "$running" ]]; then
        local n
        n=$(echo "$running" | wc -l | tr -d ' ')
        log_info "Leaving $n running agent-server container(s) alone (sibling batch or active scoring)"
    fi

    # NOTE: do NOT call `docker container prune -f` here. That command
    # removes every stopped container on the host, including unrelated
    # ones. The targeted `docker rm` above already cleans up our
    # stopped containers; disk reclamation for unrelated tenants is not
    # our job.

    # Wait for Docker to stabilize
    sleep 2
}

kill_ghost_containers() {
    # Kill containers that have been running longer than RUN_TIMEOUT.
    # Under cross-task parallelism, sibling tasks' containers may be alive
    # for legitimate reasons — skip this preemptive sweep entirely. The
    # per-attempt `timeout` already enforces the per-run cap.
    if [[ "$MAX_PARALLEL_TASKS" -gt 1 ]]; then
        return 0
    fi
    local long_runners
    long_runners=$(docker ps --filter "ancestor=${DOCKER_IMAGE}" --format '{{.ID}} {{.RunningFor}}' 2>/dev/null || true)
    if [[ -n "$long_runners" ]]; then
        # Threshold scales with RUN_TIMEOUT: anything running longer than
        # ~RUN_TIMEOUT minutes is a ghost from a prior crashed run.
        local cutoff_minutes=$(( RUN_TIMEOUT / 60 ))
        [[ $cutoff_minutes -lt 5 ]] && cutoff_minutes=5
        while IFS= read -r line; do
            local cid
            cid=$(echo "$line" | awk '{print $1}')
            # docker's RunningFor strings: "N seconds/minutes/hours/days ago"
            # Match any "hour|day" or any minute-count >= cutoff_minutes.
            if echo "$line" | grep -qE "(hour|day)"; then
                docker rm -f "$cid" 2>/dev/null || true
                log_warn "Killed ghost container: $cid"
                continue
            fi
            local mins
            mins=$(echo "$line" | grep -oE '[0-9]+ minute' | head -1 | awk '{print $1}')
            if [[ -n "$mins" && "$mins" -ge "$cutoff_minutes" ]]; then
                docker rm -f "$cid" 2>/dev/null || true
                log_warn "Killed ghost container: $cid (age ${mins} min)"
            fi
        done <<< "$long_runners"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# TASK DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

discover_tasks() {
    if [[ -n "$SPECIFIC_TASKS" ]]; then
        echo "$SPECIFIC_TASKS" | tr ',' '\n'
    else
        # All directories in dataset/ that contain instruction.md
        for d in "${DATASET_DIR}"/*/; do
            if [[ -f "${d}instruction.md" ]]; then
                basename "$d"
            fi
        done | sort
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIGS
# ─────────────────────────────────────────────────────────────────────────────

get_model_configs() {
    if [[ -n "$SPECIFIC_MODELS" ]]; then
        echo "$SPECIFIC_MODELS" | tr ',' '\n'
    else
        echo "opus"
        echo "gpt"
        echo "gemini"
    fi
}

model_to_config() {
    local model="$1"
    case "$model" in
        opus)   echo "$OPUS_CONFIG" ;;
        gpt)    echo "$GPT_CONFIG" ;;
        gemini) echo "$GEMINI_CONFIG" ;;
        *)      echo "$model" ;;  # Allow direct config path
    esac
}

model_to_slug() {
    local model="$1"
    case "$model" in
        opus)   echo "claude-opus" ;;
        gpt)    echo "gpt5.5" ;;
        gemini) echo "gemini-3.1" ;;
        *)      echo "$model" ;;
    esac
}

# Substring of the inference output-dir name to pass via `goku-eval --models`.
# Source eval_outputs dirs use the OpenHands convention
# `<display_name>_sdk_<sha>_maxiter_<N>`, where display_name comes from the
# LLM config's filename stem (e.g. `.llm_config/claude-opus-4.7.json`
# -> dir starts with `claude-opus-4.7_sdk_...`). Derived dynamically from
# the auto-discovered config paths so the slug always matches the actual
# dir name regardless of what annotators name their configs.
model_to_export_slug() {
    local model="$1"
    local cfg
    cfg=$(model_to_config "$model")
    if [[ -z "$cfg" ]]; then
        echo "$model"
        return
    fi
    basename "$cfg" .json
}

# ─────────────────────────────────────────────────────────────────────────────
# RESUME LOGIC
# ─────────────────────────────────────────────────────────────────────────────

is_run_complete() {
    local task="$1"
    local model="$2"
    local run_num="$3"

    # Standard OpenHands layout: ${OUTPUT_BASE}/run_${run_num}/run_1/goku/<display>_sdk_<sha>_maxiter_<N>/<task>/scores.jsonl
    local search_dir="${OUTPUT_BASE}/run_${run_num}/run_1/goku"
    if [[ ! -d "$search_dir" ]]; then
        return 1
    fi
    # Derive pattern from the LLM config filename stem (matches inference
    # dir naming). Falls back to legacy patterns so we can still resume
    # against older eval_outputs/ data produced by pre-display_name runs.
    local export_slug
    export_slug=$(model_to_export_slug "$model")
    local legacy_pattern
    case "$model" in
        opus)   legacy_pattern="bedrock_converse_arn" ;;
        gpt)    legacy_pattern="openai_gpt" ;;
        gemini) legacy_pattern="gemini_gemini" ;;
        *)      legacy_pattern="$model" ;;
    esac
    # Anchor both patterns to a model-dir component that must be followed
    # by `_sdk_` (the canonical OpenHands output-dir suffix). The previous
    # pattern `/${legacy_pattern}` was unanchored — a stale legacy slug
    # appearing anywhere in the path (e.g. as a substring of a sibling
    # task) would falsely flag the run as complete, causing `--runs N` to
    # silently short-circuit before producing fresh inference.
    if find "$search_dir" -path "*/${task}/scores.jsonl" 2>/dev/null \
        | grep -qE "/(${export_slug}|${legacy_pattern})_sdk_"; then
        return 0
    fi
    return 1
}

count_completed_runs() {
    local task="$1"
    local model="$2"
    local count=0
    for run_num in $(seq 1 "$RUNS_PER_MODEL"); do
        if is_run_complete "$task" "$model" "$run_num"; then
            count=$((count + 1))
        fi
    done
    echo "$count"
}

# ─────────────────────────────────────────────────────────────────────────────
# RUN A SINGLE MODEL ON A TASK (all runs)
# ─────────────────────────────────────────────────────────────────────────────

run_model_on_task() {
    local task="$1"
    local model="$2"
    local model_config
    local model_slug
    model_config=$(model_to_config "$model")
    model_slug=$(model_to_slug "$model")
    
    local model_output="${OUTPUT_BASE}"
    # Aggregated log for the whole (task, model) — operator-facing index.
    local model_log="${LOG_DIR}/${task}_${model_slug}.log"

    log_info "[${task}] Starting ${model_slug} (${RUNS_PER_MODEL} runs)"
    
    # Check how many runs already complete
    local completed
    completed=$(count_completed_runs "$task" "$model")
    if [[ "$completed" -ge "$RUNS_PER_MODEL" ]]; then
        log_ok "[${task}] ${model_slug} — all ${RUNS_PER_MODEL} runs already complete, skipping"
        return 0
    fi
    
    # Determine which runs to do
    local runs_to_do=()
    for run_num in $(seq 1 "$RUNS_PER_MODEL"); do
        if ! is_run_complete "$task" "$model" "$run_num"; then
            runs_to_do+=("$run_num")
        fi
    done
    
    log_info "[${task}] ${model_slug} — ${#runs_to_do[@]} runs needed (${completed} already done)"
    
    for run_num in "${runs_to_do[@]}"; do
        local attempt=0
        local success=false
        
        while [[ $attempt -lt $MAX_RETRIES_PER_RUN ]] && [[ "$success" == "false" ]]; do
            attempt=$((attempt + 1))
            log_info "[${task}] ${model_slug} run ${run_num} — attempt ${attempt}/${MAX_RETRIES_PER_RUN}"

            # Per-attempt log file. The aggregate `$model_log` collects the
            # full operator-facing transcript, but inference-output for THIS
            # attempt also goes to a unique file so the post-run "No instances
            # to process" grep can only see this attempt's text. Without
            # this isolation, the grep matches stale text from attempt N-1
            # and mis-classifies a fresh failure as a harness-skip.
            local attempt_log="${LOG_DIR}/${task}_${model_slug}_run${run_num}_attempt${attempt}.log"

            # Kill any leftover containers before this run
            kill_ghost_containers

            if [[ "$DRY_RUN" == "true" ]]; then
                log_info "[DRY-RUN] Would execute: uv run goku-infer ${model_config} --tasks-dir dataset --task ${task} --runs 1 --workspace docker --max-iterations ${MAX_ITERATIONS} --num-workers ${NUM_WORKERS} --output-dir ${model_output}/run_${run_num} --critic pass --judge-llm-config ${JUDGE_CONFIG}"
                success=true
                continue
            fi

            # Run with timeout. tee to BOTH the per-attempt log (used by
            # the grep below) and the aggregate (operator inspection).
            local run_start
            run_start=$(date +%s)

            if timeout "$RUN_TIMEOUT" uv run goku-infer "$model_config" \
                --tasks-dir dataset \
                --task "$task" \
                --runs 1 \
                --workspace docker \
                --max-iterations "$MAX_ITERATIONS" \
                --num-workers "$NUM_WORKERS" \
                --output-dir "${model_output}/run_${run_num}" \
                --critic pass \
                --judge-llm-config "$JUDGE_CONFIG" \
                2>&1 | tee "$attempt_log" >> "$model_log"; then

                local run_end
                run_end=$(date +%s)
                local duration=$((run_end - run_start))

                # Verify scores.jsonl was produced
                if is_run_complete "$task" "$model" "$run_num"; then
                    log_ok "[${task}] ${model_slug} run ${run_num} — COMPLETE (${duration}s)"
                    success=true
                else
                    # Distinguish silent-skip (harness filter) from honest failure.
                    # Grep ONLY the per-attempt log so stale text from a
                    # previous attempt can't cause a false-positive match.
                    if grep -q "No instances to process" "$attempt_log" 2>/dev/null; then
                        log_error "[${task}] ${model_slug} run ${run_num} — HARNESS SKIPPED (${duration}s): \"No instances to process\""
                        log_error "  This is the resume-state filter blocking re-inference."
                        log_error "  Fix: re-run this batch with --rerun (strips output.jsonl + output.critic_attempt_*.jsonl)"
                    else
                        log_warn "[${task}] ${model_slug} run ${run_num} — finished but no scores.jsonl (${duration}s)"
                    fi
                fi
            else
                local exit_code=$?
                if [[ $exit_code -eq 124 ]]; then
                    log_error "[${task}] ${model_slug} run ${run_num} — TIMEOUT after ${RUN_TIMEOUT}s"
                else
                    log_error "[${task}] ${model_slug} run ${run_num} — FAILED (exit=${exit_code})"
                fi
                
                # Force cleanup after failure
                cleanup_docker
                sleep "$CONTAINER_STARTUP_WAIT"
            fi
        done
        
        if [[ "$success" == "false" ]]; then
            log_error "[${task}] ${model_slug} run ${run_num} — EXHAUSTED retries, moving on"
        fi
    done
    
    # Final completed count
    completed=$(count_completed_runs "$task" "$model")
    log_info "[${task}] ${model_slug} — ${completed}/${RUNS_PER_MODEL} runs complete"
}

# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL EXECUTION PER TASK
# ─────────────────────────────────────────────────────────────────────────────

run_task() {
    local task="$1"
    
    log_info "═══════════════════════════════════════════════════════"
    log_info "TASK: ${task}"
    log_info "═══════════════════════════════════════════════════════"
    
    # Pre-task Docker cleanup. Skip under cross-task parallelism — sibling
    # tasks may have legitimately running containers we must not touch.
    if [[ "$MAX_PARALLEL_TASKS" -le 1 ]]; then
        cleanup_docker
        sleep "$CONTAINER_STARTUP_WAIT"
    fi
    
    # Get models to run
    local models=()
    while IFS= read -r m; do
        models+=("$m")
    done < <(get_model_configs)
    
    if [[ "$MAX_PARALLEL_MODELS" -le 1 ]]; then
        # Sequential
        for model in "${models[@]}"; do
            run_model_on_task "$task" "$model"
        done
    else
        # Parallel — run models concurrently
        local pids=()
        local model_names=()
        
        for model in "${models[@]}"; do
            run_model_on_task "$task" "$model" &
            pids+=($!)
            model_names+=("$model")
            
            # Throttle: don't exceed MAX_PARALLEL_MODELS
            if [[ ${#pids[@]} -ge $MAX_PARALLEL_MODELS ]]; then
                # Wait for any one to finish
                wait -n "${pids[@]}" 2>/dev/null || true
                # Remove completed PIDs
                local new_pids=()
                local new_names=()
                for i in "${!pids[@]}"; do
                    if kill -0 "${pids[$i]}" 2>/dev/null; then
                        new_pids+=("${pids[$i]}")
                        new_names+=("${model_names[$i]}")
                    fi
                done
                pids=("${new_pids[@]}")
                model_names=("${new_names[@]}")
            fi
        done
        
        # Wait for all remaining
        for pid in "${pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
    fi
    
    # Post-task Docker cleanup (prevent container buildup). Skip under
    # cross-task parallelism — would kill sibling tasks' containers.
    if [[ "$MAX_PARALLEL_TASKS" -le 1 ]]; then
        cleanup_docker
    fi

    log_info "TASK ${task} — ALL MODELS DONE"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# DELIVERY EXPORT
# ─────────────────────────────────────────────────────────────────────────────

export_delivery() {
    if [[ "$SKIP_EXPORT" == "true" ]]; then
        log_info "Skipping delivery export (--skip-export)"
        return 0
    fi
    
    log_info "Generating delivery export..."

    # Build --models as separate args (argparse nargs="+" requires
    # space-separated values). Each value must be a substring of the
    # inference output-dir name so the export filter matches; clean
    # delivery folder names come from get_model_display_name() and the
    # display-name swap happens inside export_delivery_format.
    local model_args=()
    while IFS= read -r m; do
        model_args+=("$(model_to_export_slug "$m")")
    done < <(get_model_configs)

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would export delivery to ${DELIVERY_DIR} with models: ${model_args[*]}"
        return 0
    fi

    uv run goku-eval \
        --output-dir "$OUTPUT_BASE" \
        --models "${model_args[@]}" \
        --runs "$RUNS_PER_MODEL" \
        --tasks-dir dataset \
        --export-delivery "$DELIVERY_DIR" \
        >> "$BATCH_LOG" 2>&1 || log_warn "Delivery export had warnings (check log)"
    
    log_ok "Delivery exported to: ${DELIVERY_DIR}"
}

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

print_summary() {
    log_info ""
    log_info "═══════════════════════════════════════════════════════"
    log_info "BATCH RUN SUMMARY"
    log_info "═══════════════════════════════════════════════════════"
    
    local total_tasks=0
    local total_complete=0
    local total_failed=0
    
    while IFS= read -r task; do
        total_tasks=$((total_tasks + 1))
        local task_ok=true
        
        while IFS= read -r model; do
            local completed
            completed=$(count_completed_runs "$task" "$model")
            local slug
            slug=$(model_to_slug "$model")
            
            if [[ "$completed" -ge "$RUNS_PER_MODEL" ]]; then
                log_info "  ✓ ${task} / ${slug}: ${completed}/${RUNS_PER_MODEL} runs"
            else
                log_warn "  ✗ ${task} / ${slug}: ${completed}/${RUNS_PER_MODEL} runs"
                task_ok=false
            fi
        done < <(get_model_configs)
        
        if [[ "$task_ok" == "true" ]]; then
            total_complete=$((total_complete + 1))
        else
            total_failed=$((total_failed + 1))
        fi
    done < <(discover_tasks)
    
    log_info ""
    log_info "Total tasks: ${total_tasks}"
    log_ok   "Fully complete: ${total_complete}"
    if [[ $total_failed -gt 0 ]]; then
        log_error "Incomplete: ${total_failed}"
    fi
    log_info "Output: ${OUTPUT_BASE}"
    log_info "Logs: ${LOG_DIR}"
    log_info "Batch log: ${BATCH_LOG}"
    log_info "═══════════════════════════════════════════════════════"

    # Export the summary counts so the caller (main) can set a non-zero
    # exit code if nothing actually ran. Without this, a batch where every
    # task silently failed via the "No instances to process" filter still
    # reported BATCH COMPLETE with exit 0, masking real failure.
    BATCH_TOTAL_TASKS=$total_tasks
    BATCH_FULLY_COMPLETE=$total_complete
    BATCH_INCOMPLETE=$total_failed
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

main() {
    log_info "═══════════════════════════════════════════════════════"
    log_info "GOKU BATCH RUNNER — Batch ${BATCH_ID}"
    log_info "═══════════════════════════════════════════════════════"
    log_info "Dataset: ${DATASET_DIR}"
    log_info "Output:  ${OUTPUT_BASE}"
    log_info "Models:  $(get_model_configs | tr '\n' ', ')"
    log_info "Runs:    ${RUNS_PER_MODEL} per model per task"
    log_info "Timeout: ${RUN_TIMEOUT}s per run"
    log_info "Parallel: ${MAX_PARALLEL_MODELS} models concurrently"
    log_info "Dry-run: ${DRY_RUN}"
    log_info ""
    
    # Verify prerequisites
    if ! command -v docker &>/dev/null; then
        log_error "Docker not found in PATH"
        exit 1
    fi
    if ! docker info &>/dev/null; then
        log_error "Docker daemon not running"
        exit 1
    fi
    if ! command -v uv &>/dev/null; then
        log_error "uv not found in PATH"
        exit 1
    fi
    
    # Verify model configs exist (auto-discovered from $LLM_CONFIG_DIR)
    local _avail
    _avail=$(ls "$LLM_CONFIG_DIR"/*.json 2>/dev/null | tr '\n' ' ' || true)
    for model in $(get_model_configs); do
        local config
        config=$(model_to_config "$model")
        if [[ -z "$config" || ! -f "$config" ]]; then
            log_error "No LLM config discovered for model alias: '${model}'"
            log_error "  Looked in: ${LLM_CONFIG_DIR}/*.json"
            log_error "  Discovered files: ${_avail:-(none)}"
            log_error "  Configs are classified by their 'model' field — opus/sonnet/claude → opus role, gpt/openai → gpt role, gemini → gemini role."
            log_error "  Override explicitly: export OPUS_CONFIG=/path/your-opus.json (or GPT_CONFIG/GEMINI_CONFIG)"
            exit 1
        fi
    done
    if [[ -z "$JUDGE_CONFIG" || ! -f "$JUDGE_CONFIG" ]]; then
        log_error "No judge LLM config discovered (classifier looks for 'kimi'/'moonshot' in model field or 'judge' in filename)"
        log_error "  Looked in: ${LLM_CONFIG_DIR}/*.json"
        log_error "  Discovered files: ${_avail:-(none)}"
        log_error "  Override explicitly: export JUDGE_CONFIG=/path/your-judge.json"
        exit 1
    fi
    
    # Verify Docker image exists
    if ! docker image inspect "$DOCKER_IMAGE" &>/dev/null; then
        log_warn "Docker image not found locally, pulling: ${DOCKER_IMAGE}"
        docker pull "$DOCKER_IMAGE" || { log_error "Failed to pull image"; exit 1; }
    fi
    
    # Discover tasks
    local tasks=()
    while IFS= read -r t; do
        tasks+=("$t")
    done < <(discover_tasks)
    
    if [[ ${#tasks[@]} -eq 0 ]]; then
        log_error "No tasks found in ${DATASET_DIR}"
        exit 1
    fi
    
    log_info "Found ${#tasks[@]} task(s): ${tasks[*]}"
    log_info ""

    # --rerun: strip prior resume-state for the targeted tasks so the
    # OpenHands eval harness actually re-runs them. Without this, even
    # after re-inferring is the intent, the harness sees the task in
    # output.jsonl and/or output.critic_attempt_*.jsonl and silently
    # skips it ("No instances to process"), causing a 0-min batch.
    if [[ "$RERUN" == "true" && ${#tasks[@]} -gt 0 ]]; then
        local task_csv
        task_csv=$(IFS=,; echo "${tasks[*]}")
        # Forward --models too, so `--rerun --tasks X --models gpt` clears
        # state ONLY for the gpt model dir, not for opus + gemini as well.
        # Without this, a prompt-edit re-run on one model would force the
        # other two to re-infer needlessly, wasting hours of compute.
        local models_arg=()
        if [[ -n "$SPECIFIC_MODELS" ]]; then
            models_arg=(--models "$SPECIFIC_MODELS")
        fi
        log_info "--rerun: cleaning resume-state for: ${task_csv}${SPECIFIC_MODELS:+ (models=${SPECIFIC_MODELS})}"
        if [[ "$DRY_RUN" == "true" ]]; then
            uv run python -m benchmarks.goku.scripts.clean_resume_state \
                --output-base "$OUTPUT_BASE" \
                --tasks "$task_csv" \
                ${models_arg[@]+"${models_arg[@]}"} \
                --dry-run \
                >> "$BATCH_LOG" 2>&1 \
                || log_warn "clean_resume_state (dry-run) reported issues; see batch log"
        else
            uv run python -m benchmarks.goku.scripts.clean_resume_state \
                --output-base "$OUTPUT_BASE" \
                --tasks "$task_csv" \
                ${models_arg[@]+"${models_arg[@]}"} \
                >> "$BATCH_LOG" 2>&1 \
                || { log_error "clean_resume_state failed; aborting"; exit 1; }
        fi
        log_ok "--rerun cleanup complete"
    fi

    # Initial cleanup
    cleanup_docker

    # Run each task
    local batch_start
    batch_start=$(date +%s)
    
    if [[ "$MAX_PARALLEL_TASKS" -le 1 ]]; then
        # Sequential tasks (default — safest for memory-constrained Docker)
        for task in "${tasks[@]}"; do
            run_task "$task"
        done
    else
        # Cross-task parallelism — spawn up to MAX_PARALLEL_TASKS tasks at
        # once. Each task itself fans out to MAX_PARALLEL_MODELS containers,
        # so concurrent containers ≈ MAX_PARALLEL_TASKS × MAX_PARALLEL_MODELS.
        log_info "Cross-task parallelism enabled: up to ${MAX_PARALLEL_TASKS} tasks × ${MAX_PARALLEL_MODELS} models = ${task_cap:=$((MAX_PARALLEL_TASKS * MAX_PARALLEL_MODELS))} concurrent containers."
        # One-time docker cleanup before launching — sibling tasks will
        # respect each other's containers thereafter.
        cleanup_docker
        sleep "$CONTAINER_STARTUP_WAIT"

        local task_pids=()
        for task in "${tasks[@]}"; do
            run_task "$task" &
            task_pids+=($!)

            # Bound concurrency: when we hit the cap, reap any completed
            # task processes before spawning more. Spin in a small loop
            # (not just `wait -n`) so we drain ALL exited children, not
            # just one — otherwise simultaneous exits can push us over.
            while [[ ${#task_pids[@]} -ge $MAX_PARALLEL_TASKS ]]; do
                wait -n "${task_pids[@]}" 2>/dev/null || true
                local live_pids=()
                for p in "${task_pids[@]}"; do
                    if kill -0 "$p" 2>/dev/null; then
                        live_pids+=("$p")
                    fi
                done
                task_pids=("${live_pids[@]}")
                # If wait -n returned but no PID was actually reaped (race
                # with last child), don't busy-loop. Short sleep keeps CPU
                # idle while we wait for the next exit.
                [[ ${#task_pids[@]} -ge $MAX_PARALLEL_TASKS ]] && sleep 1
            done
        done

        # Drain remaining task processes
        for pid in "${task_pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
    fi
    
    local batch_end
    batch_end=$(date +%s)
    local batch_duration=$(( (batch_end - batch_start) / 60 ))
    
    log_info "Total batch time: ${batch_duration} minutes"

    # Export delivery
    export_delivery

    # Print summary (sets BATCH_TOTAL_TASKS/BATCH_FULLY_COMPLETE/BATCH_INCOMPLETE)
    BATCH_TOTAL_TASKS=0
    BATCH_FULLY_COMPLETE=0
    BATCH_INCOMPLETE=0
    print_summary

    # Distinguish three completion modes so callers (and watchers) can react:
    #   exit 0  → at least one task fully complete
    #   exit 2  → nothing ran at all (silent-failure mode we hit on Modal 500s
    #             and on the output.critic_attempt_1.jsonl filter regression)
    #   exit 1  → some tasks ran but at least one missed runs
    if [[ "$DRY_RUN" == "true" ]]; then
        log_ok "BATCH COMPLETE (dry-run) — ${batch_duration} minutes total"
        return 0
    fi
    if [[ "$BATCH_FULLY_COMPLETE" -eq 0 && "$BATCH_TOTAL_TASKS" -gt 0 ]]; then
        log_error "BATCH FAILED — 0 of ${BATCH_TOTAL_TASKS} tasks completed in ${batch_duration} minutes."
        log_error "  Common causes:"
        log_error "    1. Stale resume-state — try re-running with --rerun"
        log_error "    2. Docker / Modal agent-server connectivity issues"
        log_error "    3. LLM config / Bedrock credential rejection"
        log_error "  Inspect per-model logs in ${LOG_DIR}/${task}_*.log"
        exit 2
    fi
    if [[ "$BATCH_INCOMPLETE" -gt 0 ]]; then
        log_warn "BATCH PARTIAL — ${BATCH_FULLY_COMPLETE} of ${BATCH_TOTAL_TASKS} tasks fully complete, ${BATCH_INCOMPLETE} incomplete (${batch_duration} min)"
        exit 1
    fi
    log_ok "BATCH COMPLETE — ${batch_duration} minutes total"
}

# Run
main "$@"
