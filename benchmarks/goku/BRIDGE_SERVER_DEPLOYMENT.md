# Deploying goku's Claude Code & Codex subscription bridges on a server

**The complete, single-source guide** to running goku's subscription-billed
trajectory generation on a headless Linux server (e.g. AWS EC2):

- **Claude Code** (`--agent-backend claudecode`) → Claude Pro/Max subscription
- **OpenAI Codex / gpt-5.5** (`--codex-subscription`) → ChatGPT Pro/Team subscription

Instead of paying per-token API pricing, each agent's LLM calls are routed through
a small local **bridge** that swaps a stub key for your subscription's OAuth token.
This guide takes you from a blank EC2 box to a verified, running deployment, with
the **exact command and expected output at every step**. Follow it top to bottom.

---

## Contents

1. [The two hard parts of a server deploy](#1-the-two-hard-parts-of-a-server-deploy)
2. [How it works (architecture)](#2-how-it-works-architecture)
3. [How goku consumes each bridge — READ FIRST](#3-how-goku-consumes-each-bridge--read-first)
4. [Step 1 — Provision the EC2 instance](#4-step-1--provision-the-ec2-instance)
5. [Step 2 — Install the toolchain](#5-step-2--install-the-toolchain)
6. [Step 3 — Get your subscription credentials onto the server (Option A: login on server · Option B: copy from Mac)](#6-step-3--get-your-subscription-credentials-onto-the-server)
7. [Step 4 — Copy credentials to the server (Option B only)](#7-step-4--copy-credentials-to-the-server-option-b-only)
8. [Step 5 — Verify credentials load](#8-step-5--verify-credentials-load)
9. [Step 6 — Fix Linux Docker networking (critical)](#9-step-6--fix-linux-docker-networking-critical)
10. [Step 7 — Bridge deep-dive: start standalone & live-test](#10-step-7--bridge-deep-dive-start-standalone--live-test)
11. [Step 8 — Run goku on the server](#11-step-8--run-goku-on-the-server)
12. [Step 9 — (Optional) persistent systemd bridge services](#12-step-9--optional-persistent-systemd-bridge-services)
13. [Token lifetime & auto-refresh](#13-token-lifetime--auto-refresh)
14. [Multi-account pools (scaling)](#14-multi-account-pools-scaling)
15. [End-to-end smoke test](#15-end-to-end-smoke-test)
16. [Troubleshooting](#16-troubleshooting)
17. [Security & ToS](#17-security--tos)
18. [Quick reference & checklist](#18-quick-reference--checklist)

---

## 1. The two hard parts of a server deploy

Everything else is standard; these two are what actually break a fresh server:

1. **The subscription login needs a browser, and a headless server has none.**
   `claude login` / `codex login` do a browser OAuth flow. You never hand-create
   credentials — the login stores the token itself. You have two ways to get that
   token onto the server (Step 3): **Option A** — log in *directly on the server*
   (the login writes the token there; the browser step goes through an SSH tunnel),
   or **Option B** — log in on your Mac and copy the token file over. Both end with
   the same token file the bridge reads; the long-lived refresh token then renews it
   on the server automatically.
2. **Linux Docker networking is different from Docker Desktop.** On your Mac the
   in-container agent reaches the host bridge via `host.docker.internal`
   automatically. On Linux it does **not** — you must bind the bridge to `0.0.0.0`
   and point the container at the **docker bridge gateway IP** (usually
   `172.17.0.1`). Both are one-env-var changes (Step 6).

---

## 2. How it works (architecture)

```
                    Server host (Linux)
 ┌───────────────────────────────────────────────────────────────┐
 │  goku-infer  (auto-starts the bridge as a subprocess)         │
 │     │                                                         │
 │     ├─ Claude bridge  :8765  (reads ~/.claude/.credentials)  │──▶ api.anthropic.com/v1/messages
 │     └─ Codex  bridge  :8788  (reads ~/.codex/auth.json)      │──▶ chatgpt.com/backend-api/codex/responses
 │            ▲  bound on 0.0.0.0                                │
 │            │  container reaches host at http://172.17.0.1:PORT│
 │  ┌─────────┴───────────────┐                                 │
 │  │ agent-server container  │  (OpenHands agent + tools /     │
 │  │  litellm / claude CLI   │   or the claude CLI for Claude) │
 │  └─────────────────────────┘                                 │
 └───────────────────────────────────────────────────────────────┘
```

Each bridge is a local FastAPI proxy:

| | Claude bridge | Codex bridge |
|---|---|---|
| Package | `benchmarks/utils/claude_oauth/` | `benchmarks/utils/openai_codex/` |
| API shape | Anthropic Messages (`/v1/messages`) | OpenAI (`/v1/chat/completions`, `/v1/responses`) |
| Token file | `~/.claude/.credentials.json` | `~/.codex/auth.json` |
| Upstream | `api.anthropic.com` | `chatgpt.com/backend-api/codex/responses` |
| Default port | 8765 | 8788 |
| Client base URL env | `ANTHROPIC_API_BASE` / `ANTHROPIC_BASE_URL` | `OPENAI_BASE_URL` |
| Bridge secret env | `WCB_CC_BRIDGE_SECRET` | `KAIJU_CODEX_BRIDGE_SECRET` |
| Refresh endpoint (auto) | `console.anthropic.com/v1/oauth/token` | `auth.openai.com/oauth/token` |

The client talks to the bridge with a stub key; the bridge swaps it for the real
OAuth bearer, adds the provider-specific headers, and forwards upstream.

---

## 3. How goku consumes each bridge — READ FIRST

There are **two ways** a bridge is used, and they differ per provider. Knowing
which you're doing avoids confusion later.

| | **Auto-start (default)** | **Standalone service** |
|---|---|---|
| Who starts the bridge | `goku-infer` starts it as a subprocess for the run | You run it yourself (systemd/tmux); it stays up across runs |
| **Claude** (`--agent-backend claudecode`) | ✅ **This is the path.** goku starts its own bridge. | A standalone Claude bridge is **not** consumed by the claudecode backend (goku starts its own). Useful only for verification or to serve other Anthropic clients. |
| **Codex** (`--codex-subscription`) | ✅ goku starts its own bridge. | ✅ **Alternative:** run a standalone Codex bridge and point an LLM config's `base_url` at it; run goku **without** `--codex-subscription`. Good for many jobs sharing one bridge. |

**Bottom line:** for a normal deploy you need, for both providers: credentials in
place (Steps 3–5) + networking env (Step 6). goku auto-starts the bridge during the
run. The standalone start (Step 7) is used to **verify** the whole path before
involving goku, and the systemd service (Step 9) is an optional persistent setup.

---

## 4. Step 1 — Provision the EC2 instance

- **AMI**: Ubuntu 22.04/24.04 LTS (x86_64 or arm64 — match your Docker images).
- **Type**: at least `t3.xlarge` / `m6i.xlarge` (4 vCPU, 16 GB); more for parallel workers.
- **Disk**: 60–100 GB gp3 (Docker images + task media + eval outputs).
- **Security group (firewall)**:
  - **Inbound**: only SSH (22) from your IP. **Do NOT open the bridge ports (8765/8788).**
  - **Outbound**: allow HTTPS (443) — the bridges must reach `api.anthropic.com`,
    `chatgpt.com`, `console.anthropic.com`, `auth.openai.com`, your judge provider
    (Bedrock/Gemini/OpenAI), and Docker/GHCR/PyPI.
- **IAM role** (only if you use a Bedrock judge): attach a role with
  `bedrock:InvokeModel`, or plan to pass judge keys via env.

---

## 5. Step 2 — Install the toolchain

> **Use Docker's official install, NOT `apt-get install docker.io`.** The Ubuntu
> `docker.io` package does **not** include the **buildx** plugin, and goku's first
> run **builds the agent-server image with `docker buildx`** — with only
> `docker.io` that build fails. The official install bundles buildx + compose.

```bash
# Docker (official — includes buildx + compose)
curl -fsSL https://get.docker.com | sudo sh
sudo systemctl enable --now docker
sudo usermod -aG docker $USER          # log out/in (or `newgrp docker`) so docker works w/o sudo
docker buildx version                   # MUST succeed — confirms buildx is present

# uv (Python) + git/jq
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL -l
sudo apt-get update && sudo apt-get install -y git jq

# goku
git clone <your-goku-remote> goku && cd goku
git submodule update --init --recursive     # vendored OpenHands SDK (REQUIRED — uv sync fails without it)
uv sync                                       # installs deps incl. fastapi/uvicorn for the bridges
```

> **First run builds a large Docker image.** The very first `goku-infer` run
> compiles the OpenHands agent-server image from the Node/Python base
> (`docker buildx`, several minutes, needs outbound internet to PyPI + the base
> image). It's cached afterward. To pre-build, do a throwaway 1-task run once.

---

## 6. Step 3 — Get your subscription credentials onto the server

The bridge needs your subscription's OAuth token in a place it can read
(`~/.claude/.credentials.json` and `~/.codex/auth.json` on Linux). **You never
hand-craft these — a login writes them.** There are two ways to get there; pick one:

- **Option A — log in directly on the server** (no file copy). Cleanest; the login
  writes the token file *on the server*. The one catch is the browser step needs an
  SSH tunnel on a headless box. → **6a**
- **Option B — log in on your Mac, copy the token to the server.** Most reliable /
  no tunnel: the login already produced the token, you just move it. → **6b + Step 4**

> Either way the end state is identical: the CLI's login stores the OAuth token in
> the file above and the bridge reads it. You are **not** creating credentials by
> hand — this is the same "log in and it uses your subscription" flow, just placed
> on a machine that can't run a browser itself.

### 6a. Option A — log in directly on the server (no file copy)

> ⚠️ **Not yet verified end-to-end on a headless box.** The mechanism is standard
> OAuth-over-SSH-tunnel and is safe, but the exact callback port is CLI-specific —
> read it from each login's output as you go. If it fights you, fall back to Option
> B (6b), which is the path we've actually run.

**Install the CLIs on the server** (they need Node):
```bash
# On the server
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g @anthropic-ai/claude-code @openai/codex
```

**Log in with an SSH tunnel for the OAuth callback.** Each login starts a local
callback listener (`http://localhost:<PORT>`) and prints an auth URL; you forward
that port so your laptop browser's redirect reaches the server:

```bash
# Codex — Terminal 1 (on the server):
codex login
#   → note the "listening on localhost:<PORT>" line + the auth URL it prints
# Terminal 2 (on your LAPTOP) — forward that PORT to the server:
ssh -N -L <PORT>:localhost:<PORT> ubuntu@<server>
#   → then open the auth URL from Terminal 1 in your laptop browser, sign in with
#     ChatGPT. The redirect to localhost:<PORT> tunnels to the server, and the
#     login writes ~/.codex/auth.json ON THE SERVER.

# Claude — same pattern:
claude          # then type: /login   (same browser + localhost:<PORT> callback)
# Terminal 2 on your laptop: ssh -N -L <PORT>:localhost:<PORT> ubuntu@<server>
#   → sign in; the login writes ~/.claude/.credentials.json ON THE SERVER
#     (Linux has no Keychain, so it's a file — created by the login, not by you).
```

If a CLI offers a "paste this code / paste the redirect URL back" device flow, use
that — **no tunnel needed.** After login succeeds, **skip Step 4** and go straight
to Step 5 — the credentials already live on the server.

### 6b. Option B — log in on your Mac, then copy (proven, no tunnel)

You already logged in on your Mac, which stored the token (Claude → **Keychain**,
Codex → `~/.codex/auth.json`). Export/copy that token; Step 4 moves it to the
server. It carries a long-lived refresh token, so the bridge refreshes on the
server indefinitely after.

**Claude Code:**
```bash
# On your Mac — dump the existing Keychain token to a file so it can be transferred:
security find-generic-password -s "Claude Code-credentials" -w > claude_creds.json
python3 -c "import json;d=json.load(open('claude_creds.json'));o=d['claudeAiOauth'];print('plan:',o['subscriptionType'],'| token:',o['accessToken'][:15],'| has_refresh:',bool(o.get('refreshToken')))"
# Expected: plan: max | token: sk-ant-oat01-... | has_refresh: True
```

**Codex (ChatGPT):**
```bash
# On your Mac, where `codex login` already created this file:
cp ~/.codex/auth.json codex_auth.json
python3 -c "import json;d=json.load(open('codex_auth.json'));t=d['tokens'];print('mode:',d.get('auth_mode'),'| account:',t['account_id'][:8],'| has_refresh:',bool(t.get('refresh_token')))"
# Expected: mode: chatgpt | account: ... | has_refresh: True   (mode MUST be chatgpt)
```

---

## 7. Step 4 — Copy credentials to the server (Option B only)

> **Skip this entire step if you used Option A** (login on the server) — the
> credentials already live on the server. This step is only for Option B.

Put each file at the bridge's default path.

```bash
# Claude → ~/.claude/.credentials.json
scp claude_creds.json ubuntu@<server>:/tmp/claude_creds.json
ssh ubuntu@<server> 'mkdir -p ~/.claude && mv /tmp/claude_creds.json ~/.claude/.credentials.json && chmod 600 ~/.claude/.credentials.json && ls -l ~/.claude/.credentials.json'

# Codex → ~/.codex/auth.json
scp codex_auth.json ubuntu@<server>:/tmp/codex_auth.json
ssh ubuntu@<server> 'mkdir -p ~/.codex && mv /tmp/codex_auth.json ~/.codex/auth.json && chmod 600 ~/.codex/auth.json && ls -l ~/.codex/auth.json'
```

> **Alternatives** (per bridge, precedence high→low):
> - Claude: `CLAUDE_CODE_CREDENTIALS='<json>'` (inline) → `WCB_CC_CREDS_PATH=/path` → `~/.claude/.credentials.json`
> - Codex: `CODEX_CREDENTIALS='<json>'` (inline) → `KAIJU_CODEX_AUTH_PATH=/path` → `~/.codex/auth.json`
>
> Inline env is handy for CI (no file on disk); the trade-off is there's no file
> to persist a rotated refresh token back to (the long-lived refresh token stays
> valid on its own, so it still works — just re-export periodically on long-lived boxes).

---

## 8. Step 5 — Verify credentials load

Before anything else, confirm each bridge can load (and refresh) its token:

```bash
cd ~/goku
uv run python -m benchmarks.utils.claude_oauth --check
# → [bridge] credentials OK (token prefix: sk-ant-oat01-...)

uv run python -m benchmarks.utils.openai_codex --check
# → [codex-bridge] credentials OK (token prefix: eyJhbGci..., account: ...)
```

(A warning that the bridge secret is unset is normal here — `--check` doesn't need
it.) If either fails, fix the credential file/path before continuing — every run
would otherwise fail on a dead upstream.

---

## 9. Step 6 — Fix Linux Docker networking (critical)

On Docker Desktop the container reaches the host via `host.docker.internal`. On
**Linux this does not resolve**, and a bridge bound to `127.0.0.1` is unreachable
from the container. Two changes fix it for both bridges:

1. **Bind the bridge to all interfaces** (`0.0.0.0`) so the container can reach it.
2. **Point the container at the docker bridge gateway IP** instead of `host.docker.internal`.

Why (verified against the code): goku's workspace container is run with
`docker run -d ... -p <host_port>:8000 <image>` on the **default bridge** and
**without** `--add-host host.docker.internal:host-gateway`. So inside the container
`host.docker.internal` does not resolve, but the container's default gateway **is**
the docker0 bridge IP, which reaches a host service bound on `0.0.0.0`.

### 9a. Write a persistent env file (source it in every session)

Do **not** just `export` in one shell — a fresh `tmux` session won't inherit it,
the alias will be empty, and the bridge falls back to `host.docker.internal` and
fails. Write a file and source it:

```bash
GW=$(docker network inspect bridge -f '{{range .IPAM.Config}}{{.Gateway}}{{end}}')  # usually 172.17.0.1
cat > ~/goku/bridge.env <<EOF
# Linux/server bridge networking — source this before every goku-infer run
export GOKU_CC_BRIDGE_BIND=0.0.0.0
export GOKU_CC_BRIDGE_HOST_ALIAS=$GW
export GOKU_CODEX_BRIDGE_BIND=0.0.0.0
export GOKU_CODEX_BRIDGE_HOST_ALIAS=$GW
# pinned bridge secrets (recommended — reproducible logs; a shared systemd bridge needs a fixed value)
export GOKU_CC_BRIDGE_SECRET=$(openssl rand -hex 24)      # goku launcher passes this as WCB_CC_BRIDGE_SECRET
export KAIJU_CODEX_BRIDGE_SECRET=$(openssl rand -hex 24)
# judge credentials (uncomment what you use)
# export GEMINI_API_KEY=...
# export AWS_BEARER_TOKEN_BEDROCK=... ; export AWS_REGION_NAME=us-east-1
EOF
echo "wrote ~/goku/bridge.env (gateway=$GW)"
```

Then `source ~/goku/bridge.env` in every session (or append it to `~/.bashrc`).

### 9b. Confirm the container can actually reach the host (do this once)

A host firewall (`ufw`) is the usual culprit that silently blocks docker0 → host:

```bash
source ~/goku/bridge.env
python3 -m http.server 9999 --bind 0.0.0.0 &  HTTP_PID=$!
docker run --rm curlimages/curl:latest -s -o /dev/null -w "reach=%{http_code}\n" \
  http://$GOKU_CC_BRIDGE_HOST_ALIAS:9999/     # expect: reach=200
kill $HTTP_PID
```

If it prints `reach=200`, the container↔host path works. If it hangs/fails, the
host firewall is blocking it — allow just the docker bridge:

```bash
sudo ufw allow in on docker0        # (or iptables: sudo iptables -I INPUT -i docker0 -j ACCEPT)
```

---

## 10. Step 7 — Bridge deep-dive: start standalone & live-test

This verifies the **entire** credential → OAuth → provider path **before** involving
goku. Do it once per bridge. (During a real goku run you do NOT start these by hand
— goku auto-starts its own; see Step 8.)

### 10a. Claude bridge — standalone + live call

Terminal 1 — start it:
```bash
cd ~/goku
source ~/goku/bridge.env
WCB_CC_BRIDGE_SECRET=$GOKU_CC_BRIDGE_SECRET \
uv run python -m benchmarks.utils.claude_oauth --host 0.0.0.0 --port 8765
# → [bridge] credentials OK (token prefix: sk-ant-oat01-...)
# → [bridge] listening on http://0.0.0.0:8765
```

Terminal 2 — health + a real subscription call (Anthropic Messages shape):
```bash
source ~/goku/bridge.env
curl -s http://127.0.0.1:8765/healthz ; echo          # {"ok":true}

curl -s http://127.0.0.1:8765/v1/messages \
  -H "x-api-key: $GOKU_CC_BRIDGE_SECRET" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-opus-4-7","max_tokens":32,"messages":[{"role":"user","content":"Reply with exactly: CLAUDE BRIDGE OK"}]}'
```
Expected: the message call returns content `CLAUDE BRIDGE OK` (`stop_reason: end_turn`).
**401** = `x-api-key` ≠ the secret. Upstream auth error = OAuth token/refresh failed
(re-do Steps 3–5). `Ctrl-C` to stop once verified.

### 10b. Codex bridge — standalone + live call

Terminal 1:
```bash
cd ~/goku
source ~/goku/bridge.env
KAIJU_CODEX_BRIDGE_SECRET=$KAIJU_CODEX_BRIDGE_SECRET \
uv run python -m benchmarks.utils.openai_codex --host 0.0.0.0 --port 8788
# → [codex-bridge] credentials OK (token prefix: eyJhbGci..., account: ...)
# → [codex-bridge] listening on http://0.0.0.0:8788
```

Terminal 2 — health + a real subscription call (OpenAI chat shape):
```bash
source ~/goku/bridge.env
curl -s http://127.0.0.1:8788/healthz ; echo          # {"ok":true,...}

curl -s http://127.0.0.1:8788/v1/chat/completions \
  -H "Authorization: Bearer $KAIJU_CODEX_BRIDGE_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.5","stream":false,"messages":[{"role":"user","content":"Reply with exactly: CODEX BRIDGE OK"}]}'
```
Expected: a `chat.completion` whose content is `CODEX BRIDGE OK`. **401** = secret
mismatch; `400 Unsupported parameter: X` = add `X` to `KAIJU_CODEX_STRIP_PARAMS`.
`Ctrl-C` to stop.

---

## 11. Step 8 — Run goku on the server

Run inside `tmux`/`screen` so the job survives SSH disconnects. goku auto-starts the
bridge using the env from Step 6.

### 11a. Claude Code (subscription) run

```bash
cd ~/goku
tmux new -s claude
source ~/goku/bridge.env               # loads bind/alias/secret/judge env
uv run goku-infer .llm_config/cc-opus-4-7.json \
  --agent-backend claudecode --cc-model claude-opus-4-7 \
  --tasks-dir sample_tasks --task task_lst_02 \
  --num-workers 1 --cc-timeout 5400 --max-retries 0 --n-critic-runs 1 \
  --judge-llm-config .llm_config/gemini-3.5-flash.json
```

goku reads `GOKU_CC_BRIDGE_*`, auto-starts the bridge (you'll see
`Claude Code OAuth bridge ready...` in the log), runs the `claude` CLI in the
container against it, and tears it down at the end.

### 11b. Codex / gpt-5.5 (subscription) run

```bash
cd ~/goku
tmux new -s codex
source ~/goku/bridge.env
GOKU_IMAGE_MODE=inline \
uv run goku-infer .llm_config/gpt-5.5-codex.json \
  --codex-subscription \
  --tasks-dir sample_tasks --task task_lst_02 \
  --output-dir eval_outputs/gpt55_codex \
  --num-workers 1 --max-retries 0 --n-critic-runs 1 \
  --judge-llm-config .llm_config/gemini-3.5-flash.json
```

> - `GOKU_IMAGE_MODE=inline` inlines task images as base64 (avoids the S3
>   requirement OpenAI's URL-fetching path would otherwise demand for >20-image
>   tasks). For very large image sets, configure S3 (`AWS_REGION`+`AWS_BUCKET`) and
>   drop this instead.
> - **Judge credentials** are separate/metered: for a Gemini judge set
>   `GEMINI_API_KEY` (or put the key in the judge config's `api_key`); for Bedrock
>   set `AWS_BEARER_TOKEN_BEDROCK`+`AWS_REGION_NAME` (or use the instance IAM role).

Detach with `Ctrl-b d`; reattach with `tmux attach -t codex`.

---

## 12. Step 9 — (Optional) persistent systemd bridge services

Use this only if you want a **long-lived shared bridge** across many jobs. Note the
provider difference from Step 3: the **claudecode backend always starts its own
bridge**, so a systemd Claude bridge is for verification / other clients only. The
**Codex** systemd bridge, by contrast, can be consumed by goku via a `base_url`
config (run goku **without** `--codex-subscription`).

### Codex systemd service (`/etc/systemd/system/codex-bridge.service`)
```ini
[Unit]
Description=goku Codex OAuth bridge
After=network-online.target docker.service

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/goku
Environment=KAIJU_CODEX_BRIDGE_SECRET=REPLACE_WITH_YOUR_SECRET
Environment=KAIJU_CODEX_AUTH_PATH=/home/ubuntu/.codex/auth.json
ExecStart=/home/ubuntu/.local/bin/uv run python -m benchmarks.utils.openai_codex --host 0.0.0.0 --port 8788
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload && sudo systemctl enable --now codex-bridge
curl -s http://127.0.0.1:8788/healthz ; echo      # {"ok":true,...}
journalctl -u codex-bridge -f                      # live logs
```

Then run goku **without** `--codex-subscription`, pointing a config at the service:
```jsonc
// .llm_config/gpt-5.5-codex-shared.json
{ "model": "openai/gpt-5.5",
  "base_url": "http://172.17.0.1:8788",       // docker0 gateway : bridge port
  "api_key": "YOUR_KAIJU_CODEX_BRIDGE_SECRET",
  "display_name": "gpt-5.5-codex" }
```
```bash
GOKU_IMAGE_MODE=inline uv run goku-infer .llm_config/gpt-5.5-codex-shared.json \
  --tasks-dir sample_tasks --task task_lst_02 --output-dir eval_outputs/gpt55_codex \
  --num-workers 1 --max-retries 0 --n-critic-runs 1 \
  --judge-llm-config .llm_config/gemini-3.5-flash.json
```

The Claude equivalent runs `uv run python -m benchmarks.utils.claude_oauth
--host 0.0.0.0 --port 8765` with `Environment=WCB_CC_BRIDGE_SECRET=...` — but
remember the claudecode backend won't use it; for a normal Claude run just rely on
the auto-start (Step 8).

---

## 13. Token lifetime & auto-refresh

- Both bridges hold a **short-lived access token** and a **long-lived refresh
  token**. When the access token nears expiry the bridge calls the provider's OAuth
  endpoint (Claude → `console.anthropic.com`, Codex → `auth.openai.com`), gets a
  fresh token, and **writes it back to the on-disk file**. No cron needed.
- You only re-copy credentials if the **refresh token** is revoked (you logged out /
  rotated on your Mac) or the provider invalidates it (rare, months).
- With **inline env** creds (`CLAUDE_CODE_CREDENTIALS` / `CODEX_CREDENTIALS`) there
  is no file to write back to — the bridge just refreshes from the still-valid
  refresh token each start. Fine for CI; prefer the file form on a standing server.

---

## 14. Multi-account pools (scaling)

One subscription has a rate/quota ceiling. Both bridges round-robin over several
accounts, cooling down an account on a `429` cap (each teammate logs in on their own
machine and exports their token):

```bash
# Codex — colon-separated auth.json paths ("default" = ~/.codex/auth.json)
export KAIJU_CODEX_ACCOUNT_POOL="default:/home/ubuntu/creds/acct2.json:/home/ubuntu/creds/acct3.json"
export KAIJU_CODEX_POOL_STATE_PATH=/home/ubuntu/creds/codex_pool_state.json   # cooldowns survive restarts

# Claude — colon-separated credential files
export WCB_CC_ACCOUNT_POOL="/home/ubuntu/creds/claude1.json:/home/ubuntu/creds/claude2.json"
```

`GET /quota` on either bridge shows per-account status. Keep each account under
~1 req/sec to avoid anti-abuse locks.

---

## 15. End-to-end smoke test

Run this once after Steps 1–6 to confirm creds + upstream + secret all work before a
real batch:

```bash
cd ~/goku
source ~/goku/bridge.env
# 1) creds load
uv run python -m benchmarks.utils.claude_oauth --check
uv run python -m benchmarks.utils.openai_codex --check
# 2) one live subscription call through the Codex bridge (proves upstream)
KAIJU_CODEX_BRIDGE_SECRET=$KAIJU_CODEX_BRIDGE_SECRET uv run python - <<'PY'
import httpx, os
from benchmarks.utils.openai_codex import CodexBridge
b=CodexBridge().start()
try:
    r=httpx.post(f"{b.base_url}/v1/chat/completions",
        headers={"Authorization":f"Bearer {b.stub_api_key}"},
        json={"model":"gpt-5.5","stream":False,
              "messages":[{"role":"user","content":"Reply: BRIDGE OK"}]}, timeout=120)
    print(r.status_code, r.json()["choices"][0]["message"]["content"])
finally:
    b.stop()
PY
# 3) container↔host reachability (Step 6b), then a real 1-task run (Step 8)
```

If step 2 prints `200 BRIDGE OK`, credentials + upstream + secret all work; a real
run then only exercises the container→bridge network hop (Step 6).

---

## 16. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `--check` → `credentials error` | token file missing/wrong path, or Codex not `chatgpt` mode | Re-copy (Step 4); confirm `auth_mode: chatgpt`. |
| `healthz` 503 `ok:false` | access token expired + refresh failed | Re-mint creds on the Mac (Step 3), re-copy. |
| Live call **401** | client secret ≠ bridge secret | Use the same value for `x-api-key`/`Authorization: Bearer` and the bridge's secret env. |
| Live call upstream 401 | OAuth token dead / refresh token revoked | Re-login on the Mac, re-copy. |
| Run fails instantly, **Connection refused** to `host.docker.internal` | Linux networking not set | `GOKU_*_BRIDGE_BIND=0.0.0.0` + `GOKU_*_BRIDGE_HOST_ALIAS=<docker0 gw>` (Step 6); confirm you `source`d `bridge.env`. |
| Container reachability test fails / hangs | host firewall blocks docker0 → host | `sudo ufw allow in on docker0` (Step 6b). |
| First-run build fails (`buildx` not found) | installed `docker.io` (no buildx) | Reinstall via `get.docker.com` (Step 2). |
| Codex `400 Unsupported parameter: X` | codex backend rejects param X | `export KAIJU_CODEX_STRIP_PARAMS=X` (comma-sep for more). |
| `Task has N images (>20) ... requires S3` | many-image task, OpenAI URL path | Set `GOKU_IMAGE_MODE=inline` (or configure `AWS_REGION`+`AWS_BUCKET`). |
| Judge fails `Missing Gemini API key` | judge creds not set | Export `GEMINI_API_KEY` (or Bedrock `AWS_BEARER_TOKEN_BEDROCK`+`AWS_REGION_NAME`). |
| `429` / cap | quota hit | Wait for reset, or add accounts to the pool (Step 14). |
| Job dies on SSH disconnect | ran in the foreground | Use `tmux`/`screen`. |
| Orphaned bridge after hard kill | parent SIGKILLed (atexit skipped) | `pkill -f 'benchmarks.utils.(openai_codex|claude_oauth)'`. |

---

## 17. Security & ToS

- **Credentials are live OAuth secrets.** `chmod 600` the token files, keep them off
  git (`~/.claude`, `~/.codex`, any `creds/` dir), and restrict SSH to your IP.
- **Always set the bridge secret.** Without `WCB_CC_BRIDGE_SECRET` /
  `KAIJU_CODEX_BRIDGE_SECRET` the bridge is unauthenticated and any local process
  can spend your subscription (it logs a warning at startup).
- **Never open bridge ports (8765/8788) in the AWS security group.** Bind `0.0.0.0`
  only because Linux containers need it; keep the port host/docker-local. If a host
  firewall blocks the container, allow only the docker bridge (`ufw allow in on
  docker0`), never external hosts.
- **ToS gray zone.** Running a Claude or ChatGPT *subscription* through an automated
  bridge for batched benchmark generation is against the spirit of both providers'
  acceptable-use policies; accounts have been suspended for similar patterns. Use
  for research/evaluation, keep request rates modest, and for production use the
  metered API-key path (a normal API-key LLM config, no `--codex-subscription` /
  `--agent-backend claudecode`).

---

## 18. Quick reference & checklist

### Env-var reference

| Purpose | Claude bridge | Codex bridge |
|---|---|---|
| Creds (file, default) | `~/.claude/.credentials.json` | `~/.codex/auth.json` (`auth_mode: chatgpt`) |
| Creds (custom path) | `WCB_CC_CREDS_PATH` | `KAIJU_CODEX_AUTH_PATH` |
| Creds (inline) | `CLAUDE_CODE_CREDENTIALS` | `CODEX_CREDENTIALS` |
| Bridge secret (bridge process) | `WCB_CC_BRIDGE_SECRET` | `KAIJU_CODEX_BRIDGE_SECRET` |
| Bridge secret (goku launcher) | `GOKU_CC_BRIDGE_SECRET` | `KAIJU_CODEX_BRIDGE_SECRET` |
| Bind address (goku launcher) | `GOKU_CC_BRIDGE_BIND=0.0.0.0` | `GOKU_CODEX_BRIDGE_BIND=0.0.0.0` |
| Container host alias (goku launcher) | `GOKU_CC_BRIDGE_HOST_ALIAS=172.17.0.1` | `GOKU_CODEX_BRIDGE_HOST_ALIAS=172.17.0.1` |
| Strip extra params | — | `KAIJU_CODEX_STRIP_PARAMS` |
| Force model name | — | `KAIJU_CODEX_MODEL` |
| Multi-account pool | `WCB_CC_ACCOUNT_POOL` | `KAIJU_CODEX_ACCOUNT_POOL` / `KAIJU_CODEX_POOL_STATE_PATH` |
| Client base URL | `ANTHROPIC_API_BASE` / `ANTHROPIC_BASE_URL` | `OPENAI_BASE_URL` |
| Default port | 8765 | 8788 |
| goku flag | `--agent-backend claudecode --cc-model ...` | `--codex-subscription` |

### Deploy checklist

**One-time server setup**
- [ ] Step 1 instance provisioned (SSH-only inbound, HTTPS outbound)
- [ ] Step 2 Docker via `get.docker.com` (`docker buildx version` works), uv, goku (`submodule update` + `uv sync`)
- [ ] Step 6 `~/goku/bridge.env` written (gateway + `0.0.0.0` + secrets); container↔host reach test = 200

**Claude bridge**
- [ ] Step 3 credentials on server → `~/.claude/.credentials.json` (600): Option A (login on server) **or** Option B (login on Mac → Step 4 copy)
- [ ] Step 5 `--check` → `credentials OK`
- [ ] Step 7a standalone `/healthz` + live `CLAUDE BRIDGE OK`
- [ ] Step 8a real run with `--agent-backend claudecode`

**Codex bridge**
- [ ] Step 3 credentials on server → `~/.codex/auth.json` (600, `auth_mode: chatgpt`): Option A **or** Option B (→ Step 4 copy)
- [ ] Step 5 `--check` → `credentials OK`
- [ ] Step 7b standalone `/healthz` + live `CODEX BRIDGE OK`
- [ ] Step 8b real run with `--codex-subscription` (or Step 9 systemd + `base_url` config)
