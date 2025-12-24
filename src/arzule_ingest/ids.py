"""ID generation utilities for traces and spans (W3C-compatible hex IDs)."""

from __future__ import annotations

import secrets


def new_trace_id() -> str:
    """Generate a new 32-character hex trace ID (128-bit)."""
    return secrets.token_hex(16)


def new_span_id() -> str:
    """Generate a new 16-character hex span ID (64-bit)."""
    return secrets.token_hex(8)


def new_run_id() -> str:
    """Generate a new UUID-style run ID."""
    import uuid

    return str(uuid.uuid4())

