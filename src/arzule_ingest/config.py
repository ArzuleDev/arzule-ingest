"""Configuration for Arzule ingestion wrapper."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ArzuleConfig:
    """Configuration loaded from environment or explicit values."""

    tenant_id: str
    project_id: str
    api_key: Optional[str] = field(default=None)
    ingest_url: Optional[str] = field(default=None)

    # Batching configuration
    batch_size: int = 100
    flush_interval_seconds: float = 5.0

    # Redaction toggles
    redact_enabled: bool = True
    redact_pii: bool = True  # SOC2: PII redaction enabled by default

    # SOC2 compliance settings
    require_tls: bool = True  # Enforce HTTPS for HTTP sink
    audit_log_enabled: bool = True  # Enable audit logging

    # Payload size limits
    max_inline_payload_bytes: int = 64 * 1024  # 64KB
    max_value_chars: int = 20_000

    @classmethod
    def from_env(cls, prefix: str = "ARZULE_") -> "ArzuleConfig":
        """
        Load configuration from environment variables.

        Args:
            prefix: Environment variable prefix (default: "ARZULE_")
        """

        def _get(name: str) -> Optional[str]:
            return os.environ.get(prefix + name) or os.environ.get(name)

        tenant_id = _get("TENANT_ID") or ""
        project_id = _get("PROJECT_ID") or ""

        if not tenant_id or not project_id:
            raise ValueError(
                "TENANT_ID and PROJECT_ID must be set (or provided explicitly)."
            )

        return cls(
            tenant_id=tenant_id,
            project_id=project_id,
            api_key=_get("API_KEY"),
            ingest_url=_get("INGEST_URL"),
            batch_size=int(_get("BATCH_SIZE") or "100"),
            flush_interval_seconds=float(_get("FLUSH_INTERVAL") or "5.0"),
            redact_enabled=(_get("REDACT_ENABLED") or "true").lower()
            in {"1", "true", "yes", "y"},
            redact_pii=(_get("REDACT_PII") or "true").lower()  # SOC2: default true
            in {"1", "true", "yes", "y"},
            require_tls=(_get("REQUIRE_TLS") or "true").lower()  # SOC2: enforce TLS
            in {"1", "true", "yes", "y"},
            audit_log_enabled=(_get("AUDIT_LOG_ENABLED") or "true").lower()
            in {"1", "true", "yes", "y"},
            max_inline_payload_bytes=int(
                _get("MAX_INLINE_BYTES") or str(64 * 1024)
            ),
            max_value_chars=int(_get("MAX_VALUE_CHARS") or "20000"),
        )

