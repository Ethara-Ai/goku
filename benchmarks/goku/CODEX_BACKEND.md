# OpenAI Codex subscription mode (gpt-5.5 via ChatGPT subscription)

Goku can generate trajectories with **`gpt-5.5` billed against a ChatGPT
Pro/Team subscription** instead of a metered OpenAI API key, using the vendored
Codex OAuth bridge (`benchmarks/utils/openai_codex/`). This is the OpenAI cousin
of the Claude Code subscription path (`benchmarks/utils/claude_oauth/`).

Unlike the Claude path (which runs the `claude` CLI in the container), this uses
goku's **existing OpenHands agent** driving `openai/gpt-5.5`, with litellm
pointed at the bridge. No new agent backend, no in-container CLI.

## How it works

```
run_infer.main()  --codex-subscription        host
  └─ CodexBridge().start()                     starts an OpenAI-compatible FastAPI
       reads ~/.codex/auth.json (ChatGPT sub)   proxy on 127.0.0.1:<port> that swaps
       overrides llm.base_url / llm.api_key      the stub key for the OAuth bearer +
                                                 ChatGPT-Account-Id + codex headers,
                                                 translates Chat<->Responses, and
                                                 forwards to chatgpt.com/backend-api/
                                                 codex/responses.

evaluate_instance()  (standard OpenHands backend, unchanged)
  └─ agent LLM = openai/gpt-5.5,
     base_url = http://host.docker.internal:<port>, api_key = <bridge secret>
     → in-container litellm POSTs /v1/chat/completions to the host bridge
     → bridge translates + bills the ChatGPT subscription
  → goku's normal deterministic + LLM-judge scoring runs unchanged
```

### The translation layer (important)

The vendored `translate.py` was written for a text/edit-block pipeline and would
have **dropped tool calls and images**. goku's version is enhanced to preserve:

- assistant `tool_calls` → Responses `function_call` items (request)
- `role:"tool"` results → `function_call_output` items (request)
- user `image_url` parts → Responses `input_image` parts (request) — goku is multimodal
- `function_call` output items → Chat `tool_calls` in the **streaming** response

Without these, the OpenHands agent (native tool calling + streaming) could not act.
Covered by `benchmarks/goku/tests/test_codex_translate.py`.

## Prerequisites

1. **`codex login` on this host** (writes `~/.codex/auth.json`):
   ```bash
   npm install -g @openai/codex   # or: brew install codex
   codex login                    # sign in with a ChatGPT Pro/Team account
   ```
   Verify the bridge can load it:
   ```bash
   uv run python -m benchmarks.utils.openai_codex --check
   # [codex-bridge] credentials OK (token prefix: ..., account: ...)
   ```
2. **Docker** running (goku's workspace container reaches the host bridge via
   `host.docker.internal`).
3. Deps: `fastapi`, `uvicorn`, `httpx` (already in `pyproject.toml`).

## Usage

```bash
uv run goku-infer .llm_config/gpt-5.5-codex.json \
  --codex-subscription \
  --tasks-dir sample_tasks \
  --task task_lst_05 \
  --num-workers 1 \
  --max-retries 0 \
  --n-critic-runs 1 \
  --judge-llm-config .llm_config/gemini-3.5-flash.json
```

- `--codex-subscription` starts the bridge and overrides the config's
  `base_url`/`api_key` at runtime (the values in `gpt-5.5-codex.json` are
  placeholders).
- The **agent** is `openai/gpt-5.5`; the **judge** is whatever you pass to
  `--judge-llm-config` (independent, metered).

## Flags & env vars

| Flag / env | Default | Meaning |
|---|---|---|
| `--codex-subscription` | off | Route the agent LLM through the Codex bridge. |
| `KAIJU_CODEX_BRIDGE_SECRET` | (random) | Pin the bridge shared secret. |
| `GOKU_CODEX_BRIDGE_BIND` | `127.0.0.1` | Host bind address (Linux: `0.0.0.0`). |
| `KAIJU_CODEX_AUTH_PATH` | `~/.codex/auth.json` | Override token location. |
| `KAIJU_CODEX_ACCOUNT_POOL` | — | Colon-separated auth.json paths for multi-account round-robin. |
| `KAIJU_CODEX_MODEL` | — | Force the upstream model name (else date suffixes are stripped). |

## Networking note (Linux hosts)

Docker Desktop (macOS/Windows) reaches the loopback bridge via
`host.docker.internal` out of the box. On native Linux set
`GOKU_CODEX_BRIDGE_BIND=0.0.0.0` and run containers with
`--add-host host.docker.internal:host-gateway`. The bridge always requires the
shared secret, so binding broadly does not expose your subscription.

## ToS

Using a ChatGPT subscription for automated batch trajectory generation is a
**gray zone** of OpenAI's Acceptable Use Policy (Codex is intended for
interactive coding, not batched benchmark runs) — some community projects doing
this have had accounts suspended. Use for research/evaluation, not sustained
production; for production prefer the metered API key path (drop
`--codex-subscription`).
