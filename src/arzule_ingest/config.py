"""Configuration for Arzule ingestion wrapper."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from .mode import ArzuleMode, AuthType


@dataclass(frozen=True)
class ArzuleConfig:
    """Configuration loaded from environment or explicit values."""

    # Mode configuration (NEW)
    mode: ArzuleMode = ArzuleMode.CLOUD

    # Identifiers (optional in non-cloud modes)
    tenant_id: Optional[str] = None
    project_id: Optional[str] = None
    api_key: Optional[str] = field(default=None)
    ingest_url: Optional[str] = field(default=None)

    # Self-hosted mode settings (NEW)
    auth_type: AuthType = AuthType.BEARER
    auth_value: Optional[str] = field(default=None)

    # Local mode settings (NEW)
    output_path: Optional[str] = field(default=None)

    # Multi-mode destinations (NEW)
    destinations: Optional[list[dict[str, Any]]] = field(default=None)

    # Batching configuration
    batch_size: int = 100
    flush_interval_seconds: float = 5.0

    # Redaction toggles
    redact_enabled: bool = True
    redact_pii: bool = True  # SOC2: PII redaction enabled by default

    # SOC2 compliance settings
    require_tls: bool = True  # Enforce HTTPS for HTTP sink
    audit_log_enabled: bool = True  # Enable audit logging

    # Privacy/consent settings
    trace_collection_enabled: bool = True  # If False, traces are not sent to backend

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

        def _get_bool(name: str, default: str = "true") -> bool:
            return (_get(name) or default).lower() in {"1", "true", "yes", "y"}

        # Detect mode from environment
        from .mode import detect_mode

        mode = detect_mode()

        # Load identifiers (required validation depends on mode)
        tenant_id = _get("TENANT_ID")
        project_id = _get("PROJECT_ID")

        # For cloud mode, validate required fields
        if mode == ArzuleMode.CLOUD:
            if not tenant_id or not project_id:
                raise ValueError(
                    "TENANT_ID and PROJECT_ID must be set for cloud mode. "
                    "Set ARZULE_MODE=local for offline usage without credentials."
                )

        # Set defaults for non-cloud modes
        if not tenant_id:
            tenant_id = "local" if mode == ArzuleMode.LOCAL else "selfhosted"
        if not project_id:
            project_id = "default"

        # Load auth settings for selfhosted mode
        auth_type_str = _get("AUTH_TYPE") or "bearer"
        try:
            auth_type = AuthType(auth_type_str.lower())
        except ValueError:
            auth_type = AuthType.BEARER

        return cls(
            mode=mode,
            tenant_id=tenant_id,
            project_id=project_id,
            api_key=_get("API_KEY"),
            ingest_url=_get("INGEST_URL") or _get("SELFHOSTED_ENDPOINT"),
            auth_type=auth_type,
            auth_value=_get("AUTH_VALUE") or _get("API_KEY"),
            output_path=_get("OUTPUT_PATH"),
            batch_size=int(_get("BATCH_SIZE") or "100"),
            flush_interval_seconds=float(_get("FLUSH_INTERVAL") or "5.0"),
            redact_enabled=_get_bool("REDACT_ENABLED"),
            redact_pii=_get_bool("REDACT_PII"),  # SOC2: default true
            require_tls=_get_bool("REQUIRE_TLS"),  # SOC2: enforce TLS
            audit_log_enabled=_get_bool("AUDIT_LOG_ENABLED"),
            trace_collection_enabled=_get_bool("TRACE_COLLECTION_ENABLED"),
            max_inline_payload_bytes=int(_get("MAX_INLINE_BYTES") or str(64 * 1024)),
            max_value_chars=int(_get("MAX_VALUE_CHARS") or "20000"),
        )

