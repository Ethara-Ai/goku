"""Process-local default of ``follow_redirects=True`` for httpx clients.

Why this exists
---------------
The Docker agent-server (``openhands-agent-server``) returns ``307 Temporary
Redirect`` on some endpoints (notably file upload). The OpenHands SDK's
``RemoteWorkspace`` / ``AsyncRemoteWorkspace`` construct their internal
``httpx.Client`` / ``httpx.AsyncClient`` without ``follow_redirects=True``, so
uploads surface the 307 as an opaque failure instead of following the
redirect chain.

The upstream fix is two lines in
``vendor/software-agent-sdk/openhands-sdk/openhands/sdk/workspace/remote/{base,async_remote_workspace}.py``.
We deliberately do **not** ship that fix as a submodule change, because the
same submodule is pinned by sibling benchmarks (multi-swe-bench, swe-bench,
gaia, ...) and changing the shared SDK behind their back is bad form. Instead
we apply the equivalent default at runtime, scoped to Python processes started
from *this* repo via :mod:`benchmarks.utils.sitecustomize`.

Safety analysis
---------------
- Uses ``kwargs.setdefault("follow_redirects", True)``, so any caller that
  explicitly passes ``follow_redirects=False`` at client construction is
  honored unchanged.
- Per-request overrides (``client.get(url, follow_redirects=False)``) are
  unaffected — httpx applies the per-request value regardless of the
  client default.
- httpx still strips ``Authorization`` and ``Cookie`` headers across hosts
  on redirect by default, so this does not introduce credential-leak risk.
- Empirically scoped: every non-vendor ``httpx`` caller in this repo either
  hits a fixed endpoint that does not redirect (``benchmarks/utils/litellm_proxy.py``),
  references the string ``"httpx"`` without making HTTP calls
  (``benchmarks/swesmith/profiles.py``), is legacy code, or is a test that
  mocks ``httpx.post``/``httpx.get`` at the symbol level (bypassing
  ``Client.__init__`` entirely).
- Idempotent via the module-level ``_PATCHED`` sentinel; double-import is
  a no-op.
- Cheap to undo: when upstream ships ``follow_redirects=True`` by default
  (or makes it configurable on ``RemoteWorkspace``), delete this module and
  the call site in :mod:`benchmarks.utils.sitecustomize`.
"""

from __future__ import annotations


_PATCHED = False


def apply() -> bool:
    """Install the ``follow_redirects=True`` default on ``httpx.Client`` and
    ``httpx.AsyncClient``.

    Returns:
        ``True`` if the patch was newly applied this call. ``False`` if it
        was already applied, or if ``httpx`` is not importable in this
        interpreter.
    """
    global _PATCHED
    if _PATCHED:
        return False
    try:
        import httpx
    except ImportError:
        return False

    for cls_name in ("Client", "AsyncClient"):
        cls = getattr(httpx, cls_name, None)
        if cls is None:
            continue
        orig_init = cls.__init__

        def _make_patched(_orig):
            def _patched_init(self, *args, **kwargs):
                kwargs.setdefault("follow_redirects", True)
                _orig(self, *args, **kwargs)

            _patched_init.__wrapped__ = _orig  # type: ignore[attr-defined]
            return _patched_init

        cls.__init__ = _make_patched(orig_init)  # type: ignore[assignment,method-assign]

    _PATCHED = True
    return True


def is_applied() -> bool:
    """Return whether the patch has been installed in this interpreter."""
    return _PATCHED
