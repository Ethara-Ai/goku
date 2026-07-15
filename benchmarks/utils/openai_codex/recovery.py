"""Client-side transient-retry wrapper for the Codex bridge.

Wrap an agent/coder call so a transient failure (peer-closed connection,
incomplete chunked read, 5xx, momentary rate-limit) retries with backoff instead
of aborting a module. Provider-agnostic — it inspects the exception text, not any
codex-specific type — so it also works for the Claude Code path.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Optional, TypeVar


_LOG = logging.getLogger(__name__)
_T = TypeVar("_T")

# Substrings that mark a transient, worth-retrying failure.
_TRANSIENT_SIGNALS = (
    "peer closed connection", "incomplete chunked read", "connection reset",
    "connection aborted", "server disconnected", "read timeout", "timed out",
    "temporarily unavailable", "502 bad gateway", "503 service", "504 gateway",
    "overloaded", "internal server error", "remoteprotocolerror", "econnreset",
    "rate limit", "429",
    # aider-swallowed transient LLM errors surfaced by
    # agent.agents.raise_if_transient_llm_error (TransientLLMError) and the raw
    # litellm/mid-stream signals it matches, so the codex-recovery path retries
    # the module instead of losing the turn's work.
    "transientllmerror", "midstreamfallbackerror", "apiconnectionerror",
    "apitimeouterror",
)

_DEFAULT_BACKOFF = (5, 10, 20, 40, 60)


def _backoff_schedule() -> tuple[float, ...]:
    raw = os.environ.get("KAIJU_CODEX_TRANSIENT_BACKOFF", "")
    if raw:
        try:
            return tuple(float(x) for x in raw.split(",") if x.strip())
        except ValueError:
            pass
    return _DEFAULT_BACKOFF


def is_transient(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(sig in msg for sig in _TRANSIENT_SIGNALS)


def run_with_recovery(fn: Callable[..., _T], *args: Any,
                      max_retries: Optional[int] = None,
                      sleep: Callable[[float], None] = time.sleep,
                      **kwargs: Any) -> _T:
    """Call ``fn(*args, **kwargs)``, retrying transient failures with backoff.

    Non-transient exceptions propagate immediately. Retries are capped by
    ``max_retries`` (default: length of the backoff schedule).
    """
    backoff = _backoff_schedule()
    cap = max_retries if max_retries is not None else len(backoff)
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        # Only ordinary exceptions are retryable; never swallow KeyboardInterrupt /
        # SystemExit (those are BaseException, not Exception).
        except Exception as exc:  # noqa: BLE001 — we re-raise non-transient below
            if attempt >= cap or not is_transient(exc):
                raise
            wait = backoff[min(attempt, len(backoff) - 1)]
            _LOG.warning("codex: transient failure (%s); retry %d/%d in %ss",
                         str(exc)[:120], attempt + 1, cap, wait)
            sleep(wait)
            attempt += 1
