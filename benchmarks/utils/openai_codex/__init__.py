"""OpenAI Codex (ChatGPT-auth) subscription bridge for goku (vendored from
kaiju-harness).

Routes OpenAI traffic through a ChatGPT/Codex subscription (OAuth) instead of a
metered API key, mirroring benchmarks.utils.claude_oauth. See the module
docstrings in bridge.py / credentials.py and benchmarks/goku/CODEX_BACKEND.md for
setup + ToS caveats.
"""

from benchmarks.utils.openai_codex.credentials import (
    CodexCredentials,
    CredentialProvider,
    CredentialsError,
    MultiAccountCredentialProvider,
    load_account_pool,
    load_credentials,
    refresh_credentials,
)
from benchmarks.utils.openai_codex.errors import (
    ClassifiedError,
    ErrorKind,
    classify_openai_error,
    extract_retry_after,
)
from benchmarks.utils.openai_codex.launcher import CodexBridge


__all__ = [
    "CodexBridge",
    "CodexCredentials",
    "ClassifiedError",
    "CredentialProvider",
    "CredentialsError",
    "ErrorKind",
    "MultiAccountCredentialProvider",
    "classify_openai_error",
    "extract_retry_after",
    "load_account_pool",
    "load_credentials",
    "refresh_credentials",
]
