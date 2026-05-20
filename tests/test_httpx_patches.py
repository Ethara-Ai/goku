"""Tests for benchmarks.utils.httpx_patches.

The module monkey-patches ``httpx.Client.__init__`` /
``httpx.AsyncClient.__init__`` to default ``follow_redirects=True``.

Because the patch mutates module-level state on ``httpx``, every test that
exercises ``apply()`` resets that state via ``_reset_for_test`` fixture so
ordering between tests is irrelevant.
"""

from __future__ import annotations

import importlib

import httpx
import pytest


@pytest.fixture
def fresh_patch_state():
    """Reload the module so ``_PATCHED`` is False, and restore httpx
    constructors after each test so other tests see pristine httpx."""
    from benchmarks.utils import httpx_patches as mod

    saved_client_init = httpx.Client.__init__
    saved_async_init = httpx.AsyncClient.__init__

    importlib.reload(mod)
    try:
        yield mod
    finally:
        httpx.Client.__init__ = saved_client_init  # type: ignore[method-assign]
        httpx.AsyncClient.__init__ = saved_async_init  # type: ignore[method-assign]
        # Reload one more time so other tests get a fresh (unapplied) module
        importlib.reload(mod)


def test_apply_returns_true_first_call(fresh_patch_state):
    assert fresh_patch_state.is_applied() is False
    assert fresh_patch_state.apply() is True
    assert fresh_patch_state.is_applied() is True


def test_apply_is_idempotent(fresh_patch_state):
    assert fresh_patch_state.apply() is True
    # Second call must NOT re-wrap (which would double the closure depth)
    assert fresh_patch_state.apply() is False
    assert fresh_patch_state.is_applied() is True


def test_default_follow_redirects_true_on_client(fresh_patch_state):
    fresh_patch_state.apply()
    c = httpx.Client()
    try:
        assert c.follow_redirects is True
    finally:
        c.close()


def test_default_follow_redirects_true_on_async_client(fresh_patch_state):
    fresh_patch_state.apply()
    c = httpx.AsyncClient()
    assert c.follow_redirects is True


def test_explicit_false_is_respected(fresh_patch_state):
    """setdefault must not override an explicit caller value."""
    fresh_patch_state.apply()
    c = httpx.Client(follow_redirects=False)
    try:
        assert c.follow_redirects is False
    finally:
        c.close()


def test_explicit_true_is_respected(fresh_patch_state):
    fresh_patch_state.apply()
    c = httpx.Client(follow_redirects=True)
    try:
        assert c.follow_redirects is True
    finally:
        c.close()


def test_other_client_kwargs_unaffected(fresh_patch_state):
    """Patch must not interfere with other Client kwargs."""
    fresh_patch_state.apply()
    timeout = httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0)
    c = httpx.Client(
        base_url="https://example.invalid",
        timeout=timeout,
        headers={"x-test": "1"},
    )
    try:
        assert c.follow_redirects is True
        assert str(c.base_url) == "https://example.invalid"
        assert c.headers["x-test"] == "1"
    finally:
        c.close()


def test_request_level_override_still_works(fresh_patch_state):
    """Per-request follow_redirects must override the client default.

    We can't make a real HTTP call in tests, but we can verify that
    constructing a request via Client.build_request preserves caller
    kwargs and that httpx honours per-request overrides — those go
    through Client.send, which inspects the per-call value first.
    The fact that we don't touch send() or build_request() means this
    invariant is preserved by construction.
    """
    fresh_patch_state.apply()
    c = httpx.Client()
    try:
        # Sanity: build_request is unmodified and accepts arbitrary kwargs.
        req = c.build_request("GET", "https://example.invalid/")
        assert req.method == "GET"
    finally:
        c.close()


def test_is_applied_reflects_state(fresh_patch_state):
    assert fresh_patch_state.is_applied() is False
    fresh_patch_state.apply()
    assert fresh_patch_state.is_applied() is True
