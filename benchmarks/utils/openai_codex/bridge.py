"""OpenAI Codex (ChatGPT-auth) subscription bridge.

A local, OpenAI-compatible HTTP proxy that lets a harness generate trajectories on
a ChatGPT/Codex *subscription* instead of a metered API key — the OpenAI analogue
of agent/claude_code/bridge.py.

It accepts a Responses-API request on ``/responses`` (or ``/v1/responses``), swaps
the caller's stub key for a live ChatGPT OAuth bearer token, adds the headers the
codex backend requires, forwards to::

    POST https://chatgpt.com/backend-api/codex/responses

and streams the SSE response back byte-for-byte.

Verified live 2026-07-03 (Pro account): with the access token + account id from
``~/.codex/auth.json`` and the headers below, ``gpt-5.5`` returns a 200 Responses
SSE stream (``response.created`` … ``response.completed``).

Required upstream headers (all verified):
    Authorization:      Bearer <access_token>
    ChatGPT-Account-Id: <account_id>
    OpenAI-Beta:        responses=experimental
    originator:         codex_cli_rs
    session_id:         <uuid>            (per request)
    User-Agent:         codex_cli_rs/<ver>

Point your client at it::

    export OPENAI_BASE_URL=http://127.0.0.1:8788      # litellm / openai SDK honor this
    export OPENAI_API_KEY=$KAIJU_CODEX_BRIDGE_SECRET   # stub; bridge substitutes OAuth
    # model: openai/gpt-5.5  (Responses mode)

Security: set KAIJU_CODEX_BRIDGE_SECRET and give clients the same value as
OPENAI_API_KEY; otherwise the bridge is unauthenticated and any local process can
spend the subscription.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re
import uuid
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from benchmarks.utils.openai_codex import translate as _xlate
from benchmarks.utils.openai_codex.credentials import (
    CredentialProvider,
    CredentialsError,
    MultiAccountCredentialProvider,
    load_account_pool,
)
from benchmarks.utils.openai_codex.errors import classify_openai_error


_LOG = logging.getLogger(__name__)

UPSTREAM_DEFAULT = "https://chatgpt.com/backend-api/codex"
RESPONSES_PATH = "/responses"
OAUTH_BETA = "responses=experimental"
ORIGINATOR = "codex_cli_rs"
DEFAULT_USER_AGENT = "codex_cli_rs/0.142.5"

# Headers we always drop from the incoming request before re-signing (hop-by-hop
# or client-auth that we replace). Everything else is passed through.
_STRIP_REQUEST_HEADERS = {
    "host", "authorization", "content-length", "connection", "accept-encoding",
    "openai-organization", "openai-beta", "chatgpt-account-id", "originator",
    "session_id", "user-agent", "x-api-key",
    # `accept` and `content-type` are FORCED below (the codex backend rejects
    # anything but text/event-stream + application/json), so drop any inherited value.
    "accept", "content-type",
}


def _upstream_base() -> str:
    return os.environ.get("KAIJU_CODEX_UPSTREAM", UPSTREAM_DEFAULT).rstrip("/")


def _user_agent() -> str:
    return os.environ.get("KAIJU_CODEX_USER_AGENT", DEFAULT_USER_AGENT)


def _force_store_false() -> bool:
    # ChatGPT-account Codex requests are not server-stored; the codex CLI sends
    # store=false. Default to enforcing it (litellm may send store=true).
    return os.environ.get("KAIJU_CODEX_FORCE_STORE_FALSE", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _bridge_secret() -> str:
    return os.environ.get("KAIJU_CODEX_BRIDGE_SECRET", "").strip()


# A standalone SSE comment line. SSE parsers (and litellm's Responses-API stream
# reader) ignore any line beginning with ":", but forwarding it downstream resets
# the client's read-timeout clock. It is ONLY ever emitted between upstream reads
# (never mid-`data:` chunk), so it can't corrupt a real event.
_KEEPALIVE_LINE = b": keepalive\n\n"


def _keepalive_interval() -> float:
    """Seconds of upstream idle before we emit a downstream SSE keep-alive.

    Default 15s (well under a typical client read-timeout); override via
    KAIJU_CODEX_KEEPALIVE_SEC. A value <= 0 disables keep-alives.
    """
    raw = os.environ.get("KAIJU_CODEX_KEEPALIVE_SEC", "15").strip()
    try:
        return float(raw)
    except ValueError:
        return 15.0


async def _stream_with_keepalive(
    chunks: AsyncIterator[bytes],
    aclose,
    interval: float,
) -> AsyncIterator[bytes]:
    """Forward upstream ``chunks`` byte-for-byte, interleaving an SSE keep-alive
    comment whenever the upstream has been idle for ``interval`` seconds.

    Real chunks are yielded exactly as received. The keep-alive is emitted only
    when NO upstream chunk is in flight (between reads), so it can never land in
    the middle of a ``data:`` event. Stops when the upstream iterator is exhausted
    (``response.completed`` etc.) or raises; always closes the upstream response.
    Cancellation (client disconnect) propagates and the ``finally`` still closes.
    """
    ait = chunks.__aiter__()
    nxt: Optional[asyncio.Future] = None
    try:
        while True:
            nxt = asyncio.ensure_future(ait.__anext__())
            while True:
                try:
                    # Wait for the next upstream chunk, but no longer than the
                    # keep-alive interval. asyncio.shield keeps the pending
                    # __anext__ alive across a timeout so we don't drop a chunk.
                    if interval and interval > 0:
                        chunk = await asyncio.wait_for(asyncio.shield(nxt), interval)
                    else:
                        chunk = await nxt
                except asyncio.TimeoutError:
                    # Upstream idle (stalled or still reasoning): keep the client
                    # connection warm and keep waiting on the SAME pending read.
                    yield _KEEPALIVE_LINE
                    continue
                except StopAsyncIteration:
                    nxt = None
                    return
                else:
                    nxt = None
                    yield chunk
                    break
    except asyncio.CancelledError:
        # Client disconnected / server shutdown: cancel the in-flight read.
        if nxt is not None and not nxt.done():
            nxt.cancel()
        raise
    finally:
        await aclose()


def _secret_eq(candidate: str, secret: str) -> bool:
    # Constant-time compare so a local attacker can't recover the secret by timing.
    return bool(candidate) and hmac.compare_digest(candidate, secret)


def _client_authorized(request: Request) -> bool:
    secret = _bridge_secret()
    if not secret:
        return True  # unauthenticated mode (logged as a warning at startup)
    # Accept the secret via Bearer Authorization or the OpenAI x-api-key header.
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer ") and _secret_eq(auth[7:].strip(), secret):
        return True
    if _secret_eq(request.headers.get("x-api-key", "").strip(), secret):
        return True
    return False


# A trailing OpenAI date snapshot suffix, e.g. "gpt-5.5-2026-04-23".
_DATE_SUFFIX_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")

# Responses-API params the ChatGPT/Codex backend REJECTS (400 "Unsupported
# parameter"). The harness/litellm routinely send these (e.g. max_tokens ->
# max_output_tokens, sampling params on a reasoning model); strip them so the
# request is accepted. Extend via KAIJU_CODEX_STRIP_PARAMS (comma-separated).
_DEFAULT_UNSUPPORTED_PARAMS = {
    "max_output_tokens", "temperature", "top_p", "top_logprobs", "logprobs",
    "frequency_penalty", "presence_penalty", "logit_bias", "n", "seed",
    # Sent by the OpenHands SDK's Responses path; the codex backend 400s on them
    # (verified by probing the live backend: it rejects each with 400 "Unsupported
    # parameter" until all are stripped, after which gpt-5.5 returns 200).
    "prompt_cache_retention", "prompt_cache_key", "safety_identifier", "service_tier",
    "metadata", "truncation",
}


def _strip_unsupported(body: dict) -> None:
    extra = os.environ.get("KAIJU_CODEX_STRIP_PARAMS", "")
    strip = _DEFAULT_UNSUPPORTED_PARAMS | {p.strip() for p in extra.split(",") if p.strip()}
    for key in strip:
        body.pop(key, None)


def _normalize_model(body: dict) -> None:
    """Map the requested model to one the codex backend accepts.

    The ChatGPT/Codex backend accepts only bare model names (e.g. ``gpt-5.5``) and
    rejects dated snapshots (``gpt-5.5-2026-04-23`` -> 400 "model is not supported")
    that the harness's model registry uses. Strip a trailing date suffix; a full
    override is available via ``KAIJU_CODEX_MODEL``.
    """
    override = os.environ.get("KAIJU_CODEX_MODEL", "").strip()
    if override:
        body["model"] = override
        return
    model = body.get("model")
    if isinstance(model, str) and _DATE_SUFFIX_RE.search(model):
        body["model"] = _DATE_SUFFIX_RE.sub("", model)


def _normalize_input(body: dict) -> None:
    """Coerce a string `input` into the list form the codex backend requires.

    The public Responses API accepts `input` as a bare string, but the codex
    backend (chatgpt.com/backend-api/codex/responses) rejects it with
    400 "Input must be a list". Wrap a string into a single user message so
    clients that send the shorthand (e.g. litellm.responses(input="...")) work.
    """
    inp = body.get("input")
    if isinstance(inp, str):
        body["input"] = [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": inp}],
        }]


def _prepare_body(raw: bytes) -> tuple[bytes, bool]:
    """Normalize the Responses body for the codex backend.

    Returns ``(prepared_bytes, client_wanted_stream)``. The codex backend ONLY
    accepts streaming requests (400 "Stream must be set to true" otherwise), so we
    always force ``stream=true`` upstream and remember what the client asked for:
    if the client wanted a unary response we aggregate the SSE back into one JSON
    object (see ``_aggregate_sse``).

    Also: coerce a string `input` into the required list form, and enforce
    store=false when configured (ChatGPT accounts are not server-stored). Model,
    instructions, tools, reasoning otherwise pass through unchanged.
    """
    try:
        body = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return raw, False
    if not isinstance(body, dict):
        return raw, False
    client_wanted_stream = bool(body.get("stream"))
    _normalize_model(body)
    _normalize_input(body)
    _strip_unsupported(body)
    body["stream"] = True  # codex backend requires streaming
    if _force_store_false():
        body["store"] = False
    return json.dumps(body).encode(), client_wanted_stream


def _aggregate_sse(raw_sse: bytes) -> tuple[Optional[dict], Optional[str]]:
    """Collapse a Responses-API SSE stream into the final response object.

    The terminal ``response.completed`` event carries the response metadata + usage
    but an EMPTY ``output`` (the codex backend streams output incrementally). So we
    assemble ``output`` from the ``response.output_item.done`` events (each carries a
    complete item — reasoning, message, tool call — with its text) and splice it
    into the completed response. Returns ``(response_obj, error)``.
    """
    final: Optional[dict] = None
    err: Optional[str] = None
    items: list[dict] = []
    for line in raw_sse.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload in (b"", b"[DONE]"):
            continue
        try:
            evt = json.loads(payload)
        except json.JSONDecodeError:
            continue
        etype = evt.get("type", "")
        if etype == "response.output_item.done" and isinstance(evt.get("item"), dict):
            items.append(evt["item"])
        elif etype in ("response.completed", "response.incomplete") and isinstance(evt.get("response"), dict):
            final = evt["response"]
        elif etype == "response.failed":
            final = evt.get("response") if isinstance(evt.get("response"), dict) else None
            err = json.dumps(evt.get("response", evt))
        elif etype == "error":
            err = json.dumps(evt)
    if final is not None and not final.get("output") and items:
        final = {**final, "output": items}
    return final, err


def _forward_headers(request: Request, token: str, account_id: str) -> dict[str, str]:
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _STRIP_REQUEST_HEADERS
    }
    headers["Authorization"] = f"Bearer {token}"
    headers["ChatGPT-Account-Id"] = account_id
    headers["OpenAI-Beta"] = OAUTH_BETA
    headers["originator"] = ORIGINATOR
    headers["session_id"] = str(uuid.uuid4())
    headers["User-Agent"] = _user_agent()
    # FORCE these — the codex backend returns 400 "Unsupported content type" for
    # anything else (e.g. a client's default Accept: */*).
    headers["Content-Type"] = "application/json"
    headers["Accept"] = "text/event-stream"
    return headers


def _default_provider():
    """A multi-account pool if KAIJU_CODEX_ACCOUNT_POOL is set, else single-account."""
    pool_spec = os.environ.get("KAIJU_CODEX_ACCOUNT_POOL", "").strip()
    if pool_spec:
        pool = load_account_pool(pool_spec)
        if pool is not None:
            return pool
    return CredentialProvider()


def build_app(provider=None) -> FastAPI:
    """Construct the FastAPI app. `provider` is injected for tests; defaults to a
    multi-account pool (if KAIJU_CODEX_ACCOUNT_POOL is set) or a single-account
    CredentialProvider reading ~/.codex/auth.json."""
    provider = provider or _default_provider()
    app = FastAPI(title="codex-bridge")
    # Long-lived async client with generous timeouts (reasoning turns are slow).
    client = httpx.AsyncClient(timeout=httpx.Timeout(connect=15.0, read=1800.0, write=60.0, pool=15.0))

    if not _bridge_secret():
        _LOG.warning(
            "KAIJU_CODEX_BRIDGE_SECRET is not set — the bridge is UNAUTHENTICATED; "
            "any local process can spend this subscription. Set it (and point "
            "clients' OPENAI_API_KEY at the same value) to lock it down."
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:  # noqa: D401
        try:
            token = provider.get_access_token()
            return JSONResponse({"ok": True, "token_prefix": token[:12] + "...",
                                 "account_prefix": provider.account_id[:8] + "..."})
        except CredentialsError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=503)

    @app.get("/quota")
    async def quota() -> JSONResponse:  # noqa: D401
        if isinstance(provider, MultiAccountCredentialProvider):
            return JSONResponse({"multi_account": True, **provider.status()})
        return JSONResponse({"multi_account": False,
                             "account_prefix": provider.account_id[:8] + "..."})

    def _auth_or_401(request: Request):
        if not _client_authorized(request):
            return JSONResponse({"error": {"message": "bridge: unauthorized (bad OPENAI_API_KEY)",
                                           "type": "authentication_error"}}, status_code=401)
        try:
            # Atomic: token + account_id from the SAME account (multi-account safe).
            return provider.get_token_and_account()
        except CredentialsError as e:
            return JSONResponse({"error": {"message": f"bridge: {e}", "type": "credentials_error"}},
                                status_code=503)

    async def _open_upstream(request: Request, body: bytes, token: str, account: str):
        """POST the prepared body to the codex backend; return (upstream, None) or
        (None, error_Response)."""
        headers = _forward_headers(request, token, account)
        url = _upstream_base() + RESPONSES_PATH
        upstream_req = client.build_request("POST", url, content=body, headers=headers)
        try:
            upstream = await client.send(upstream_req, stream=True)
        except httpx.HTTPError as e:
            return None, JSONResponse(
                {"error": {"message": f"bridge: upstream request failed: {e}",
                           "type": "upstream_error"}}, status_code=502)
        if upstream.status_code >= 400:
            err = await upstream.aread()
            await upstream.aclose()
            classified = classify_openai_error(
                upstream.status_code, err, dict(upstream.headers))
            _LOG.warning("codex upstream %s [%s]: %s",
                         upstream.status_code, classified.kind.value, err[:300])
            # On a cap / rate-limit, cool down the active account and rotate so the
            # NEXT request uses a healthy one (multi-account pools only).
            if classified.should_rotate_account and isinstance(provider, MultiAccountCredentialProvider):
                provider.penalize(classified.retry_after or 900)
            media = upstream.headers.get("content-type", "application/json")
            return None, Response(content=err, status_code=upstream.status_code, media_type=media)
        return upstream, None

    async def _proxy_responses(request: Request) -> Response:
        auth = _auth_or_401(request)
        if isinstance(auth, Response):
            return auth
        token, account = auth

        raw = await request.body()
        body, client_wanted_stream = _prepare_body(raw)
        upstream, err_resp = await _open_upstream(request, body, token, account)
        if err_resp is not None:
            return err_resp

        if client_wanted_stream:
            media = upstream.headers.get("content-type", "text/event-stream")
            gen = _stream_with_keepalive(
                upstream.aiter_raw(), upstream.aclose, _keepalive_interval())
            return StreamingResponse(gen, status_code=200, media_type=media)

        raw_sse = await upstream.aread()
        await upstream.aclose()
        final, err = _aggregate_sse(raw_sse)
        if final is None:
            msg = err or "bridge: upstream stream produced no response.completed event"
            return JSONResponse({"error": {"message": msg, "type": "upstream_error"}}, status_code=502)
        return JSONResponse(final, status_code=200)

    async def _proxy_chat(request: Request) -> Response:
        """/chat/completions -> translate to Responses, forward, translate back.

        Lets any OpenAI Chat-Completions client (the harness preflight probe,
        litellm without responses-mode registration, the openai SDK) drive the
        codex backend, which only speaks the Responses API.
        """
        auth = _auth_or_401(request)
        if isinstance(auth, Response):
            return auth
        token, account = auth

        try:
            chat_req = json.loads(await request.body() or b"{}")
        except json.JSONDecodeError:
            return JSONResponse({"error": {"message": "bridge: invalid JSON body",
                                           "type": "invalid_request_error"}}, status_code=400)
        model = chat_req.get("model", "")
        client_wanted_stream = bool(chat_req.get("stream"))

        resp_body = _xlate.chat_to_responses(chat_req)
        # Reuse the Responses body prep (model normalization, forced stream/store).
        prepared, _ = _prepare_body(json.dumps(resp_body).encode())
        # normalized model for the echoed response
        try:
            model = json.loads(prepared).get("model", model)
        except json.JSONDecodeError:
            pass

        upstream, err_resp = await _open_upstream(request, prepared, token, account)
        if err_resp is not None:
            return err_resp

        created = _xlate.now_ts()
        if client_wanted_stream:
            # Translate the upstream Responses SSE into Chat-Completions SSE
            # INCREMENTALLY (do not buffer the whole turn) so incremental output
            # keeps a log-based liveness watchdog fed and latency stays low.
            async def _gen():
                try:
                    async for chunk in _xlate.aiter_responses_sse_as_chat(
                        upstream.aiter_lines(), model, created):
                        yield chunk
                finally:
                    await upstream.aclose()
            return StreamingResponse(_gen(), status_code=200, media_type="text/event-stream")

        raw_sse = await upstream.aread()
        await upstream.aclose()
        final, err = _aggregate_sse(raw_sse)
        if final is None:
            msg = err or "bridge: upstream stream produced no response.completed event"
            return JSONResponse({"error": {"message": msg, "type": "upstream_error"}}, status_code=502)
        return JSONResponse(_xlate.responses_to_chat(final, model, created), status_code=200)

    # Responses API (native path — preferred when the client uses responses mode).
    @app.post("/responses")
    async def responses(request: Request) -> Response:
        return await _proxy_responses(request)

    @app.post("/v1/responses")
    async def v1_responses(request: Request) -> Response:
        return await _proxy_responses(request)

    @app.post("/backend-api/codex/responses")
    async def native_responses(request: Request) -> Response:
        return await _proxy_responses(request)

    # Chat Completions API (translated to Responses) — the drop-in compatibility path.
    @app.post("/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _proxy_chat(request)

    @app.post("/v1/chat/completions")
    async def v1_chat_completions(request: Request) -> Response:
        return await _proxy_chat(request)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await client.aclose()

    return app
