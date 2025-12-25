"""Payload sanitization and redaction utilities.

SOC2 Compliance Notes:
- Secret redaction is enabled by default
- PII redaction is now enabled by default for SOC2 compliance
- All string values are scanned for sensitive patterns
"""

from __future__ import annotations

import os
import re
from typing import Any

# Keys to redact (matched case-insensitively)
REDACT_KEYS = {
    "authorization",
    "api_key",
    "apikey",
    "token",
    "access_token",
    "refresh_token",
    "password",
    "secret",
    "x-api-key",
    "bearer",
    "credential",
    "private_key",
    "client_secret",
}

# Patterns for common secrets in text
SECRET_PATTERNS = [
    re.compile(r"(api[_-]?key|apikey)\s*[:=]\s*['\"]?([^'\"\s]+)", re.IGNORECASE),
    re.compile(r"(secret|password|token|auth)\s*[:=]\s*['\"]?([^'\"\s]+)", re.IGNORECASE),
    re.compile(r"(sk-[a-zA-Z0-9]{20,})", re.IGNORECASE),  # OpenAI-style keys
    re.compile(r"(Bearer\s+[a-zA-Z0-9_\-\.]+)", re.IGNORECASE),
    re.compile(r"(ghp_[a-zA-Z0-9]{36})", re.IGNORECASE),  # GitHub PAT
    re.compile(r"(xox[baprs]-[a-zA-Z0-9-]+)", re.IGNORECASE),  # Slack tokens
    re.compile(r"(AKIA[A-Z0-9]{16})", re.IGNORECASE),  # AWS access key
]

# Common PII patterns - SOC2 requires these to be redacted by default
PII_PATTERNS = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email
    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),  # Phone (US)
    re.compile(r"\b\d{3}[-]?\d{2}[-]?\d{4}\b"),  # SSN
    re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),  # Credit card
    re.compile(r"\b\d{5}(-\d{4})?\b"),  # ZIP code
]

# Module-level config cache
_pii_redaction_enabled: bool | None = None


def _is_pii_redaction_enabled() -> bool:
    """Check if PII redaction is enabled (cached)."""
    global _pii_redaction_enabled
    if _pii_redaction_enabled is None:
        # Check environment variable, default to True for SOC2 compliance
        env_val = os.environ.get("ARZULE_REDACT_PII", "true").lower()
        _pii_redaction_enabled = env_val in {"1", "true", "yes", "y"}
    return _pii_redaction_enabled


def _truncate_str(s: str, max_chars: int) -> str:
    """Truncate string with length indicator."""
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 20)] + f"...(truncated,{len(s)} chars)"


def sanitize(
    value: Any,
    *,
    redact: bool = True,
    redact_pii_flag: bool | None = None,
    max_chars: int = 20_000,
    _depth: int = 0,
) -> Any:
    """
    Best-effort minimization/redaction for payloads.

    SOC2 Compliance:
    - Redacts known secret-bearing keys and patterns
    - Redacts PII patterns by default (emails, phones, SSNs, credit cards)
    - Truncates long strings
    - Caps recursion depth to avoid pathological structures

    Args:
        value: The value to sanitize
        redact: Whether to redact secret keys
        redact_pii_flag: Whether to redact PII. If None, reads from config/env.
        max_chars: Maximum characters for string values
        _depth: Current recursion depth (internal)

    Returns:
        Sanitized value
    """
    if _depth > 12:
        return "<max_depth>"

    # Determine PII redaction setting
    should_redact_pii = redact_pii_flag if redact_pii_flag is not None else _is_pii_redaction_enabled()

    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        result = value
        # SOC2: Apply secret pattern redaction to all strings
        if redact:
            result = redact_secrets(result)
        # SOC2: Apply PII redaction to all strings
        if should_redact_pii:
            result = redact_pii(result)
        return _truncate_str(result, max_chars)
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, (list, tuple)):
        return [
            sanitize(v, redact=redact, redact_pii_flag=should_redact_pii, max_chars=max_chars, _depth=_depth + 1)
            for v in value[:200]
        ]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in list(value.items())[:400]:
            ks = str(k)
            if redact and ks.strip().lower() in REDACT_KEYS:
                out[ks] = "<redacted>"
            else:
                out[ks] = sanitize(v, redact=redact, redact_pii_flag=should_redact_pii, max_chars=max_chars, _depth=_depth + 1)
        return out

    # Fallback: string repr, truncated and redacted
    try:
        result = repr(value)
        if redact:
            result = redact_secrets(result)
        if should_redact_pii:
            result = redact_pii(result)
        return _truncate_str(result, max_chars)
    except Exception:
        return "<unreprable>"


def redact_secrets(text: str) -> str:
    """Redact common secret patterns from text."""
    result = text
    for pattern in SECRET_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result


def redact_pii(text: str) -> str:
    """Redact common PII patterns from text."""
    result = text
    for pattern in PII_PATTERNS:
        result = pattern.sub("[PII_REDACTED]", result)
    return result


def truncate_string(s: str, max_len: int = 200) -> str:
    """Truncate a string to max length with ellipsis."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."

