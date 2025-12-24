"""Payload sanitization and redaction utilities."""

from __future__ import annotations

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
}

# Patterns for common secrets in text
SECRET_PATTERNS = [
    re.compile(r"(api[_-]?key|apikey)\s*[:=]\s*['\"]?([^'\"\s]+)", re.IGNORECASE),
    re.compile(r"(secret|password|token|auth)\s*[:=]\s*['\"]?([^'\"\s]+)", re.IGNORECASE),
    re.compile(r"(sk-[a-zA-Z0-9]{20,})", re.IGNORECASE),  # OpenAI-style keys
    re.compile(r"(Bearer\s+[a-zA-Z0-9_\-\.]+)", re.IGNORECASE),
]

# Common PII patterns
PII_PATTERNS = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email
    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),  # Phone
    re.compile(r"\b\d{3}[-]?\d{2}[-]?\d{4}\b"),  # SSN
]


def _truncate_str(s: str, max_chars: int) -> str:
    """Truncate string with length indicator."""
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 20)] + f"...(truncated,{len(s)} chars)"


def sanitize(
    value: Any,
    *,
    redact: bool = True,
    max_chars: int = 20_000,
    _depth: int = 0,
) -> Any:
    """
    Best-effort minimization/redaction for payloads.

    - Redacts known secret-bearing keys
    - Truncates long strings
    - Caps recursion depth to avoid pathological structures

    Args:
        value: The value to sanitize
        redact: Whether to redact secret keys
        max_chars: Maximum characters for string values
        _depth: Current recursion depth (internal)

    Returns:
        Sanitized value
    """
    if _depth > 12:
        return "<max_depth>"

    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_str(value, max_chars)
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, (list, tuple)):
        return [
            sanitize(v, redact=redact, max_chars=max_chars, _depth=_depth + 1)
            for v in value[:200]
        ]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in list(value.items())[:400]:
            ks = str(k)
            if redact and ks.strip().lower() in REDACT_KEYS:
                out[ks] = "<redacted>"
            else:
                out[ks] = sanitize(v, redact=redact, max_chars=max_chars, _depth=_depth + 1)
        return out

    # Fallback: string repr, truncated
    try:
        return _truncate_str(repr(value), max_chars)
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

