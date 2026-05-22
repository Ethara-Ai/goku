"""Runtime extension: add ``DocumentContent`` to the OpenHands SDK ``Message`` type.

Why this exists
---------------
The vendored OpenHands SDK only ships two content classes for ``Message.content``:
``TextContent`` and ``ImageContent``. Native PDF input is supported by the
three agent providers we use (Claude/Anthropic, OpenAI, Gemini) but each
expects a different content-block shape on the wire. We need a third content
class that:

  1. Plugs into ``Message.content``'s Pydantic-validated Sequence union.
  2. Serializes to the right native PDF block per provider when
     ``Message.to_chat_dict()`` calls each content's ``to_llm_dict()``.

We do this at runtime — never touching ``vendor/software-agent-sdk/`` —
so the shared submodule stays pristine for sibling benchmarks.

This module mirrors :mod:`benchmarks.utils.httpx_patches` in shape: a single
``apply()`` function is invoked from :mod:`benchmarks.utils.sitecustomize`.

Safety analysis
---------------
* Idempotent via the module-level ``_PATCHED`` sentinel.
* ``DocumentContent`` is a strict ``BaseContent`` subclass — old Message
  consumers that only handle Text/Image will see an unfamiliar type but
  ``to_llm_dict()`` still produces standard chat-API content blocks, so
  downstream LiteLLM serialization is unaffected for those callers.
* The patch only re-annotates ``Message.content`` and calls
  ``model_rebuild()``. No method is overridden, no instance state is
  mutated.
* Cheap to undo: when (if) the SDK upstream adds DocumentContent itself,
  delete this module and the call in ``sitecustomize.py``.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import ClassVar, Literal, Sequence


logger = logging.getLogger(__name__)


_PATCHED = False
_LITELLM_OPUS_47_PATCHED = False
DocumentContent = None  # populated by apply()

# Bedrock application-inference-profile IDs that resolve to Claude Opus 4.7
# but whose ARN carries no "opus-4-7" substring (the ID is an opaque hash).
# LiteLLM's `_is_claude_4_7_model` detects family by substring on the model
# string, so it can't see through these ARNs and falls back to the legacy
# `thinking.type=enabled` format that Bedrock Opus 4.7 rejects (HTTP 400).
#
# Extend this tuple as new profile IDs are provisioned. The env var
# ``GOKU_OPUS_47_INFERENCE_PROFILE_IDS`` (comma-separated) is merged in at
# patch time for one-off operator overrides without a code change.
_KNOWN_OPUS_47_PROFILE_IDS: tuple[str, ...] = (
    "653flds7ip4s",
)


def apply() -> bool:
    """Install ``DocumentContent`` into the SDK ``Message`` content union.

    Returns ``True`` if newly applied this call; ``False`` if already
    applied or the SDK isn't importable in this interpreter.
    """
    global _PATCHED, DocumentContent
    if _PATCHED:
        return False
    try:
        # Late import — at sitecustomize time, the SDK module is importable
        # but the import is *cheap* because we only need its base types.
        from openhands.sdk.llm import message as _msg
        from pydantic import ConfigDict
    except ImportError:
        return False

    class _DocumentContent(_msg.BaseContent):
        """Native PDF content block. Routes to a per-provider on-wire shape.

        Stored fields are deliberately minimal: ``pdf_path`` (filesystem
        reference, not embedded base64 — keeps event logs small) and
        ``provider_hint`` (selected at construction by the caller from the
        target model's identifier; see :mod:`benchmarks.goku.media_adapters`).
        """

        type: Literal["document"] = "document"
        pdf_path: str
        # 'anthropic' | 'openai' | 'gemini'.  None ⇒ fall back to OpenAI-style
        # 'file' block, which Gemini and OpenAI both accept.
        provider_hint: str | None = None

        model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

        def to_llm_dict(self) -> list[dict]:
            path = Path(self.pdf_path)
            if not path.is_file():
                logger.warning(
                    "DocumentContent.to_llm_dict: PDF missing at %s — emitting placeholder",
                    self.pdf_path,
                )
                return [{"type": "text", "text": f"[PDF missing: {self.pdf_path}]"}]
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            if self.provider_hint == "anthropic":
                block: dict = {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": b64,
                    },
                }
            else:  # 'openai' / 'gemini' / unknown → OpenAI-style file block
                block = {
                    "type": "file",
                    "file": {
                        "filename": path.name,
                        "file_data": f"data:application/pdf;base64,{b64}",
                    },
                }
            if self.cache_prompt:
                block["cache_control"] = {"type": "ephemeral"}
            return [block]

    # Re-annotate Message.content to admit DocumentContent into the union.
    # Pydantic 2 validates this Sequence union element-wise; we need to
    # rebuild the model so the new annotation is picked up.
    #
    # We use a Discriminated Union (keyed off the literal `type` field on
    # each subtype: "text" / "image" / "document"). Without a discriminator
    # pydantic tries each member in order and uses ConfigDict(extra="forbid")
    # on TextContent to short-circuit non-matches — that works but wastes
    # 2-3 validation attempts per content block AND is fragile if a future
    # ImageContent change weakens its `extra` config. The discriminated form
    # is the recommended pydantic pattern for tagged unions like this.
    from pydantic import Field
    from typing import Annotated, Union
    new_union = Sequence[
        Annotated[
            Union[_msg.TextContent, _msg.ImageContent, _DocumentContent],
            Field(discriminator="type"),
        ]
    ]
    try:
        _msg.Message.model_fields["content"].annotation = new_union
        _msg.Message.model_rebuild(force=True)
    except Exception as exc:
        logger.warning(
            "Failed to extend openhands.sdk.llm.message.Message.content with "
            "DocumentContent (%s). Native PDF on the agent side will not be "
            "available; rendering-to-images is still possible.", exc,
        )
        return False

    # Expose DocumentContent at the SDK message module so callers can do
    # `from openhands.sdk.llm.message import DocumentContent` — same import
    # surface they'd have if upstream had shipped it.
    _msg.DocumentContent = _DocumentContent  # type: ignore[attr-defined]
    DocumentContent = _DocumentContent

    _PATCHED = True

    # Independent sub-patch: teach LiteLLM how to recognize our org's opaque
    # Bedrock inference-profile ARNs as Claude Opus 4.7 so adaptive-thinking
    # routing works. Tracked on its own sentinel so a future SDK update that
    # makes the DocumentContent patch unnecessary doesn't accidentally drop
    # this fix.
    _patch_litellm_opus_47_detection()

    return True


def _patch_litellm_opus_47_detection() -> bool:
    """Make LiteLLM's adaptive-thinking detector recognize opaque Opus 4.7 ARNs.

    Why
    ---
    LiteLLM's ``AnthropicConfig._map_reasoning_effort`` emits the correct
    ``{type: "adaptive"}`` thinking block only when
    ``AnthropicModelInfo._is_claude_4_7_model(model)`` returns True. That
    detector is a substring match for ``opus-4-7``/``opus_4_7``/``opus-4.7``/
    ``opus_4.7`` on the model arg. Bedrock application-inference-profile
    ARNs (``bedrock/converse/arn:.../application-inference-profile/<hash>``)
    contain none of those markers, so detection fails and LiteLLM falls back
    to the legacy ``{type: "enabled", budget_tokens: N}`` shape — which
    Bedrock's Opus 4.7 endpoint rejects with HTTP 400.

    Fix
    ---
    Wrap ``_is_claude_4_7_model`` so it also returns True when the model
    string contains any of our known profile IDs (or any IDs supplied via
    the ``GOKU_OPUS_47_INFERENCE_PROFILE_IDS`` env var). The original
    detector is still consulted first so direct-model-id callers are
    unaffected.

    Returns ``True`` if newly applied this call, ``False`` if already
    applied or LiteLLM isn't importable.
    """
    global _LITELLM_OPUS_47_PATCHED
    if _LITELLM_OPUS_47_PATCHED:
        return False
    try:
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo
    except ImportError:
        logger.debug(
            "LiteLLM not importable; skipping Opus 4.7 inference-profile patch."
        )
        return False

    extra_ids: tuple[str, ...] = _KNOWN_OPUS_47_PROFILE_IDS
    env_override = os.getenv("GOKU_OPUS_47_INFERENCE_PROFILE_IDS", "").strip()
    if env_override:
        extra_ids = extra_ids + tuple(
            tok.strip().lower()
            for tok in env_override.split(",")
            if tok.strip()
        )

    _orig_is_47 = AnthropicModelInfo._is_claude_4_7_model

    @staticmethod
    def _is_claude_4_7_model_patched(model: str) -> bool:
        # Preserve original semantics for direct-model-id callers.
        if _orig_is_47(model):
            return True
        if not model:
            return False
        lower = model.lower()
        return any(pid in lower for pid in extra_ids)

    AnthropicModelInfo._is_claude_4_7_model = _is_claude_4_7_model_patched
    _LITELLM_OPUS_47_PATCHED = True
    logger.info(
        "Applied LiteLLM Opus 4.7 inference-profile patch (recognizes IDs: %s)",
        ", ".join(extra_ids),
    )
    return True


def is_applied() -> bool:
    """Return whether the patch has been installed in this interpreter."""
    return _PATCHED


def is_litellm_opus_47_patched() -> bool:
    """Return whether the LiteLLM Opus 4.7 detection patch is installed."""
    return _LITELLM_OPUS_47_PATCHED
