"""Host-side launcher for the Claude Code OAuth bridge.

The bridge (``benchmarks.utils.claude_oauth.bridge``) is an Anthropic-compatible
FastAPI proxy that swaps an inbound stub API key for the host's Claude Code
*subscription* OAuth token (read from macOS Keychain / ``~/.claude/.credentials.json``)
before forwarding upstream. That is what lets goku's Claude Code agent bill
trajectory generation against the Max/Pro plan instead of a metered API key.

Goku runs the agent inside a Docker workspace container, so the bridge must
listen on the host and be reachable from the container. This launcher:

  * starts ``python -m benchmarks.utils.claude_oauth`` as a subprocess bound to
    ``127.0.0.1:<port>`` (an ephemeral free port by default),
  * blocks until ``/healthz`` responds (or the process dies on a creds error),
  * exposes ``container_base_url`` = ``http://host.docker.internal:<port>`` for
    the in-container Claude Code CLI to point ``ANTHROPIC_BASE_URL`` at,
  * tears the subprocess down on context exit.

Typical use (once per eval run, shared across parallel instances)::

    with ClaudeOAuthBridge() as bridge:
        os.environ["GOKU_CC_BRIDGE_URL"] = bridge.container_base_url
        ...  # run instances; each docker-execs `claude` against the bridge
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

# Hostname a Docker Desktop / recent Docker Engine container uses to reach a
# service listening on the host loopback. Overridable for exotic networking.
DEFAULT_CONTAINER_HOST = os.environ.get("GOKU_CC_BRIDGE_HOST_ALIAS", "host.docker.internal")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _repo_root() -> Path:
    """Return the goku repo root (dir that contains the ``benchmarks`` package)
    so the ``-m benchmarks.utils.claude_oauth`` subprocess resolves imports."""
    # this file: <root>/benchmarks/utils/claude_oauth/launcher.py
    return Path(__file__).resolve().parents[3]


class ClaudeOAuthBridge:
    """Context manager owning a bridge subprocess for the lifetime of a run."""

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
        # Bind loopback by default (Docker Desktop proxies host.docker.internal
        # to the host loopback). Linux-native hosts, where the container reaches
        # the host via the docker0 gateway, need 0.0.0.0 — set GOKU_CC_BRIDGE_BIND.
        self.host = host or os.environ.get("GOKU_CC_BRIDGE_BIND", "127.0.0.1")
        self.port = port or _find_free_port()
        self.container_host = container_host
        self.startup_timeout = startup_timeout
        self.log_level = log_level
        # Shared secret locks the bridge down: only clients presenting it (as
        # ANTHROPIC_API_KEY / x-api-key) can spend the subscription. Generated
        # per run unless the caller pins one.
        self.bridge_secret = bridge_secret or os.environ.get(
            "GOKU_CC_BRIDGE_SECRET"
        ) or secrets.token_urlsafe(24)
        self._proc: subprocess.Popen | None = None

    @property
    def stub_api_key(self) -> str:
        """Value the in-container CLI must present as ANTHROPIC_API_KEY. Equal to
        the bridge secret so the request authenticates; the bridge strips it and
        forwards the real OAuth bearer upstream."""
        return self.bridge_secret

    @property
    def base_url(self) -> str:
        """Host-loopback URL (for host-side clients / preflight)."""
        return f"http://{self.host}:{self.port}"

    @property
    def container_base_url(self) -> str:
        """URL an in-container process uses to reach the bridge on the host."""
        return f"http://{self.container_host}:{self.port}"

    def preflight(self) -> None:
        """Verify subscription credentials load before starting anything.

        Runs the bridge module's ``--check`` path, which loads (and if needed
        refreshes) the OAuth token, then exits. Raises RuntimeError with the
        subprocess stderr if creds are missing/invalid — fail fast with a clear
        message instead of every instance erroring on a dead upstream.
        """
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.utils.claude_oauth", "--check"],
            cwd=str(_repo_root()),
            capture_output=True,
            text=True,
            timeout=60,
            env=self._subprocess_env(),
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Claude Code subscription credentials could not be loaded for "
                "the OAuth bridge. Log in with the `claude` CLI on this host "
                "first (or set CLAUDE_CODE_CREDENTIALS / WCB_CC_CREDS_PATH).\n"
                f"bridge --check stderr:\n{result.stderr.strip() or result.stdout.strip()}"
            )
        logger.info("Claude Code OAuth credentials preflight OK: %s", result.stdout.strip())

    def _subprocess_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        # The bridge reads WCB_CC_BRIDGE_SECRET (vendored env-var name) to
        # require the shared secret on every proxied request.
        env["WCB_CC_BRIDGE_SECRET"] = self.bridge_secret
        return env

    def start(self) -> "ClaudeOAuthBridge":
        self.preflight()
        env = self._subprocess_env()
        logger.info(
            "Starting Claude Code OAuth bridge on %s (container URL: %s)",
            self.base_url,
            self.container_base_url,
        )
        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "benchmarks.utils.claude_oauth",
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
                    f"Claude Code OAuth bridge exited during startup "
                    f"(code {self._proc.returncode}):\n{out}"
                )
            try:
                r = httpx.get(health_url, timeout=2.0)
                if r.status_code == 200:
                    logger.info("Claude Code OAuth bridge ready at %s", self.base_url)
                    return
            except Exception as exc:  # noqa: BLE001 — connection refused while booting
                last_err = exc
            time.sleep(0.4)
        self.stop()
        raise RuntimeError(
            f"Claude Code OAuth bridge did not become healthy within "
            f"{self.startup_timeout}s (last error: {last_err})"
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
        logger.info("Claude Code OAuth bridge stopped")

    def __enter__(self) -> "ClaudeOAuthBridge":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
