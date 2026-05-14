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
#   model contains "claude"/"opus"/"sonnet"           → OPUS_CONFIG
#   model contains "gpt"/"openai"/" o1"/"o3"          → GPT_CONFIG
#   model contains "gemini"                           → GEMINI_CONFIG
#   model contains "kimi"/"moonshot" or path "judge"  → JUDGE_CONFIG
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
    "JUDGE_CONFIG":  {"filename": ("judge", "kimi", "moonshot"),
                       "model":    ("kimi", "moonshot")},
    "OPUS_CONFIG":   {"filename": ("opus", "claude", "sonnet"),
                       "model":    ("claude", "opus", "sonnet")},
    "GPT_CONFIG":    {"filename": ("gpt", "openai"),
                       "model":    ("gpt", "openai")},
    "GEMINI_CONFIG": {"filename": ("gemini",),
                       "model":    ("gemini",)},
}

files = []
for path in sorted(glob.glob(os.path.join(cfg_dir, "*.json"))):
    base = os.path.basename(path).lower()
    if base in SKIP_BASENAMES or base.startswith("_"):
        continue
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        continue
    files.append((path, base, str(data.get("model", "")).lower()))

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
MAX_ITERATIONS=30
NUM_WORKERS=1              # Workers per model run (keep 1 for Docker stability)
MAX_PARALLEL_MODELS=3      # How many models to run concurrently per task
RUN_TIMEOUT=1200           # Timeout per single run in seconds (20 min)
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
        --output-dir)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --skip-export)
            SKIP_EXPORT=true
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
            echo "  --timeout SEC      Timeout per single run (default: 1200)"
            echo "  --parallel N       Models to run concurrently (default: 3)"
            echo "  --output-dir DIR   Output directory (default: eval_outputs)"
            echo "  --skip-export      Skip delivery export at end"
            echo "  --help             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

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
    log_info "Cleaning up Docker containers..."
    
    # Kill all agent-server containers (running or stopped)
    local containers
    containers=$(docker ps -a --filter "ancestor=${DOCKER_IMAGE}" -q 2>/dev/null || true)
    if [[ -n "$containers" ]]; then
        echo "$containers" | xargs docker rm -f 2>/dev/null || true
        log_info "Removed $(echo "$containers" | wc -l | tr -d ' ') agent-server containers"
    fi
    
    # Also kill any containers with "agent-server" in name (catches edge cases)
    containers=$(docker ps -a --filter "name=agent-server" -q 2>/dev/null || true)
    if [[ -n "$containers" ]]; then
        echo "$containers" | xargs docker rm -f 2>/dev/null || true
    fi
    
    # Prune stopped containers + dangling images (reclaim disk)
    docker container prune -f 2>/dev/null || true
    
    # Wait for Docker to stabilize
    sleep 2
}

kill_ghost_containers() {
    # Kill containers that have been running longer than RUN_TIMEOUT
    local long_runners
    long_runners=$(docker ps --filter "ancestor=${DOCKER_IMAGE}" --format '{{.ID}} {{.RunningFor}}' 2>/dev/null || true)
    if [[ -n "$long_runners" ]]; then
        while IFS= read -r line; do
            local cid
            cid=$(echo "$line" | awk '{print $1}')
            # If running > 30 minutes, it's a ghost
            if echo "$line" | grep -qE "(hour|day|[3-9][0-9] minute)"; then
                docker rm -f "$cid" 2>/dev/null || true
                log_warn "Killed ghost container: $cid"
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
    if find "$search_dir" -path "*/${task}/scores.jsonl" 2>/dev/null \
        | grep -qE "/${export_slug}_sdk_|/${legacy_pattern}"; then
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
            
            # Kill any leftover containers before this run
            kill_ghost_containers
            
            if [[ "$DRY_RUN" == "true" ]]; then
                log_info "[DRY-RUN] Would execute: uv run goku-infer ${model_config} --tasks-dir dataset --task ${task} --runs 1 --workspace docker --max-iterations ${MAX_ITERATIONS} --num-workers ${NUM_WORKERS} --output-dir ${model_output}/run_${run_num} --critic pass --judge-llm-config ${JUDGE_CONFIG}"
                success=true
                continue
            fi
            
            # Run with timeout
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
                >> "$model_log" 2>&1; then
                
                local run_end
                run_end=$(date +%s)
                local duration=$((run_end - run_start))
                
                # Verify scores.jsonl was produced
                if is_run_complete "$task" "$model" "$run_num"; then
                    log_ok "[${task}] ${model_slug} run ${run_num} — COMPLETE (${duration}s)"
                    success=true
                else
                    log_warn "[${task}] ${model_slug} run ${run_num} — finished but no scores.jsonl (${duration}s)"
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
    
    # Pre-task Docker cleanup
    cleanup_docker
    sleep "$CONTAINER_STARTUP_WAIT"
    
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
    
    # Post-task Docker cleanup (prevent container buildup)
    cleanup_docker
    
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
    
    # Initial cleanup
    cleanup_docker
    
    # Run each task
    local batch_start
    batch_start=$(date +%s)
    
    for task in "${tasks[@]}"; do
        run_task "$task"
    done
    
    local batch_end
    batch_end=$(date +%s)
    local batch_duration=$(( (batch_end - batch_start) / 60 ))
    
    log_info "Total batch time: ${batch_duration} minutes"
    
    # Export delivery
    export_delivery
    
    # Print summary
    print_summary
    
    log_ok "BATCH COMPLETE — ${batch_duration} minutes total"
}

# Run
main "$@"
