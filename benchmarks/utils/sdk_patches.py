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
from pathlib import Path
from typing import ClassVar, Literal, Sequence


logger = logging.getLogger(__name__)


_PATCHED = False
DocumentContent = None  # populated by apply()


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
    new_union = Sequence[
        _msg.TextContent | _msg.ImageContent | _DocumentContent  # noqa: F821
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
    return True


def is_applied() -> bool:
    """Return whether the patch has been installed in this interpreter."""
    return _PATCHED
