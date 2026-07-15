"""Host-side launcher for the OpenAI Codex (ChatGPT-subscription) bridge.

The bridge (``benchmarks.utils.openai_codex.bridge``) is an OpenAI-compatible
FastAPI proxy that swaps an inbound stub API key for the host's ChatGPT/Codex
*subscription* OAuth token (read from ``~/.codex/auth.json``) before forwarding
to ``chatgpt.com/backend-api/codex/responses``. That is what lets goku's agent
bill gpt-5.5 trajectory generation against the ChatGPT plan instead of a metered
API key.

goku runs the agent inside a Docker workspace container, so the bridge listens
on the host and is reached from the container via ``host.docker.internal``. This
launcher mirrors ``ClaudeOAuthBridge``:

  * starts ``python -m benchmarks.utils.openai_codex`` bound to ``127.0.0.1:<port>``,
  * blocks until ``/healthz`` responds (or the process dies on a creds error),
  * exposes ``container_base_url`` = ``http://host.docker.internal:<port>`` and a
    per-run ``stub_api_key`` (== the bridge secret) the client presents as
    ``OPENAI_API_KEY``,
  * tears the subprocess down on context exit.

Prerequisite: ``codex login`` must have been run on this host (writes
``~/.codex/auth.json``). Override the token location with ``KAIJU_CODEX_AUTH_PATH``.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from openhands.sdk import get_logger


logger = get_logger(__name__)

DEFAULT_CONTAINER_HOST = os.environ.get(
    "GOKU_CODEX_BRIDGE_HOST_ALIAS", "host.docker.internal"
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _repo_root() -> Path:
    # this file: <root>/benchmarks/utils/openai_codex/launcher.py
    return Path(__file__).resolve().parents[3]


class CodexBridge:
    """Context manager owning a Codex bridge subprocess for the run's lifetime."""

    def __init__(
        self,
        *,
        port: int | None = None,
        host: str | None = None,
        container_host: str = DEFAULT_CONTAINER_HOST,
        startup_timeout: float = 45.0,
        log_level: str = "info",
        bridge_secret: str | None = None,
    ) -> None:
        # Bind loopback by default (Docker Desktop proxies host.docker.internal to
        # the host loopback). Native-Linux hosts need 0.0.0.0 — set
        # GOKU_CODEX_BRIDGE_BIND.
        self.host = host or os.environ.get("GOKU_CODEX_BRIDGE_BIND", "127.0.0.1")
        self.port = port or _find_free_port()
        self.container_host = container_host
        self.startup_timeout = startup_timeout
        self.log_level = log_level
        self.bridge_secret = (
            bridge_secret
            or os.environ.get("KAIJU_CODEX_BRIDGE_SECRET")
            or secrets.token_urlsafe(24)
        )
        self._proc: subprocess.Popen | None = None

    @property
    def stub_api_key(self) -> str:
        """Value the in-container client presents as OPENAI_API_KEY (== the bridge
        secret; the bridge strips it and forwards the real OAuth bearer upstream)."""
        return self.bridge_secret

    @property
    def base_url(self) -> str:
        """Host-loopback URL (host-side clients / preflight)."""
        return f"http://{self.host}:{self.port}"

    @property
    def container_base_url(self) -> str:
        """URL an in-container process uses to reach the bridge on the host."""
        return f"http://{self.container_host}:{self.port}"

    def _subprocess_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env["KAIJU_CODEX_BRIDGE_SECRET"] = self.bridge_secret
        return env

    def preflight(self) -> None:
        """Verify ChatGPT/Codex credentials load before starting anything (runs the
        bridge module's ``--check`` path). Raises RuntimeError with clear guidance
        if ``codex login`` hasn't been done / the token is invalid."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.utils.openai_codex", "--check"],
            cwd=str(_repo_root()),
            capture_output=True,
            text=True,
            timeout=60,
            env=self._subprocess_env(),
        )
        if result.returncode != 0:
            raise RuntimeError(
                "ChatGPT/Codex subscription credentials could not be loaded for "
                "the bridge. Run `codex login` on this host first (or set "
                "CODEX_CREDENTIALS / KAIJU_CODEX_AUTH_PATH).\n"
                f"bridge --check stderr:\n{result.stderr.strip() or result.stdout.strip()}"
            )
        logger.info("Codex credentials preflight OK: %s", result.stdout.strip())

    def start(self) -> "CodexBridge":
        self.preflight()
        env = self._subprocess_env()
        logger.info(
            "Starting Codex OAuth bridge on %s (container URL: %s)",
            self.base_url,
            self.container_base_url,
        )
        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "benchmarks.utils.openai_codex",
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--log-level",
                self.log_level,
            ],
            cwd=str(_repo_root()),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._await_ready()
        return self

    def _await_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        health_url = f"{self.base_url}/healthz"
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if self._proc and self._proc.poll() is not None:
                out = self._proc.stdout.read() if self._proc.stdout else ""
                raise RuntimeError(
                    f"Codex bridge exited during startup "
                    f"(code {self._proc.returncode}):\n{out}"
                )
            try:
                r = httpx.get(health_url, timeout=2.0)
                if r.status_code == 200:
                    logger.info("Codex bridge ready at %s", self.base_url)
                    return
            except Exception as exc:  # noqa: BLE001 — connection refused while booting
                last_err = exc
            time.sleep(0.4)
        self.stop()
        raise RuntimeError(
            f"Codex bridge did not become healthy within {self.startup_timeout}s "
            f"(last error: {last_err})"
        )

    def stop(self) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
        logger.info("Codex bridge stopped")

    def __enter__(self) -> "CodexBridge":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
