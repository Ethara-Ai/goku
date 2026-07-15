# Claude Code agent backend (subscription-driven trajectory generation)

Goku can generate trajectories with the official **Claude Code CLI** instead of
the OpenHands SDK agent, billed against your **Claude Code subscription**
(Max/Pro OAuth token) rather than a metered API key. This mirrors
WildClawBench's `claudecode` backend, adapted to goku's `RemoteWorkspace` +
scoring pipeline.

## How it works

```
run_infer.main()                      host
  └─ ClaudeOAuthBridge().start()      starts an Anthropic-compatible FastAPI
       reads Keychain / ~/.claude/     proxy on 127.0.0.1:<port> that swaps the
       .credentials.json (OAuth)       stub key for your subscription bearer +
                                       oauth-2025-04-20 beta + "You are Claude
                                       Code" system prefix, then forwards to
                                       api.anthropic.com.

evaluate_instance()                   per task
  ├─ prepare_workspace()  (unchanged) creates the Docker workspace, uploads all
  │                                    task media into /workspace
  └─ _evaluate_instance_claudecode()
       ├─ ensure_cli_installed()      npm i -g @anthropic-ai/claude-code
       ├─ docker exec claude -p ...   ANTHROPIC_BASE_URL=http://host.docker
       │    --output-format             .internal:<port>, ANTHROPIC_API_KEY=<secret>
       │    stream-json --verbose      the CLI reads /workspace media from disk
       ├─ parse stream-json           → response text + trajectory + usage
       ├─ _download_outputs()         (shared) collect agent files from /workspace
       └─ _score_rubrics() +          (shared) deterministic + LLM-judge scoring,
          _persist_and_build_output() scores.jsonl, results/
```

The OpenHands backend is untouched; both share the scoring tail.

## Prerequisites

1. **Log in to Claude Code on the host** so the subscription token is available:
   ```bash
   claude            # then /login, choose your Max/Pro account
   ```
   The token is read from the macOS Keychain (`Claude Code-credentials`) or
   `~/.claude/.credentials.json` on Linux. Verify:
   ```bash
   python -m benchmarks.utils.claude_oauth --check   # prints token prefix
   ```
2. **Docker** running locally (the workspace must be a locally-backed
   `RemoteWorkspace`; API/remote runtimes are not supported by this backend).
3. Deps: `fastapi`, `uvicorn`, `httpx` (added to `pyproject.toml`).

## Usage

```bash
uv run goku-infer \
  --agent-backend claudecode \
  --cc-model opus \
  --task task_e25b6d \
  --tasks-dir tasks \
  --judge-llm-config .llm_config/gemini-3.5-flash.json
```

`--llm-config` is still required by the parser (it names the output directory
and drives the judge fallback), but the **agent** model is `--cc-model`, not the
LLM config. The judge is unchanged — it can be any Bedrock/Gemini/OpenAI model.

## Flags & env vars

| Flag / env | Default | Meaning |
|---|---|---|
| `--agent-backend claudecode` | `openhands` | Select the Claude Code backend. |
| `--cc-model` / `GOKU_CC_MODEL` | `opus` | Value passed to `claude --model` (`opus`, `sonnet`, or a full id). |
| `--cc-timeout` / `GOKU_CC_TIMEOUT` | `1800` | Per-task wall-clock cap (s) for the CLI run. |
| `--max-iterations` | (harness) | Passed to `claude --max-turns`. |
| `GOKU_CC_BRIDGE_BIND` | `127.0.0.1` | Host bind address for the bridge (see Linux note). |
| `GOKU_CC_BRIDGE_SECRET` | (random) | Pin the bridge shared secret instead of auto-generating one. |

Per-task Claude Code usage (tokens + `total_cost_usd`) is written into each
instance's `test_result` as `cc_input_tokens`, `cc_output_tokens`,
`cc_cache_read_tokens`, `cc_cache_write_tokens`, `cc_cost_usd`, `cc_num_turns`.
The raw CLI transcript is saved to `<eval_output>/<task>/claude_code_trajectory.jsonl`.

## Networking note (Linux hosts)

On Docker Desktop (macOS/Windows) the in-container CLI reaches the host bridge
via `host.docker.internal` with the bridge bound to loopback — no extra config.

On **native Linux**, `host.docker.internal` is not automatic and loopback isn't
reachable from the container. Bind the bridge to all interfaces and ensure the
container can resolve the host alias:

```bash
export GOKU_CC_BRIDGE_BIND=0.0.0.0
# and run containers with --add-host host.docker.internal:host-gateway
```

The bridge always requires the shared secret (auto-generated per run and passed
to the CLI as `ANTHROPIC_API_KEY`), so binding to `0.0.0.0` does not expose your
subscription to unauthenticated callers.

## ToS

Using a Claude Code **subscription** token for automated benchmark trajectory
generation is the same posture as WildClawBench's `claudecode` backend. Review
Anthropic's terms for your plan before running at scale.
