"""Mode detection and configuration for Arzule SDK self-hosting support."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ArzuleMode(str, Enum):
    """SDK operation modes."""

    CLOUD = "cloud"  # Arzule cloud backend (default)
    LOCAL = "local"  # Local file storage, no network
    SELFHOSTED = "selfhosted"  # Custom HTTP backend
    MULTI = "multi"  # Multiple destinations


class AuthType(str, Enum):
    """Authentication types for HTTP backends."""

    BEARER = "bearer"  # Authorization: Bearer <token>
    HEADER = "header"  # Custom header(s) from JSON
    BASIC = "basic"  # Authorization: Basic <base64(user:pass)>
    NONE = "none"  # No authentication


@dataclass
class ModeConfig:
    """Configuration for SDK mode."""

    mode: ArzuleMode
    # Identifiers (required for cloud, optional for others)
    tenant_id: str = "local"
    project_id: str = "default"
    # HTTP backend settings
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    auth_type: AuthType = AuthType.BEARER
    auth_value: Optional[str] = None
    # Local file settings
    output_path: Optional[str] = None
    # Multi-mode destinations
    destinations: Optional[list[dict[str, Any]]] = None
    # Common settings
    require_tls: bool = True
    trace_collection_enabled: bool = True


def detect_mode() -> ArzuleMode:
    """
    Auto-detect SDK mode from environment.

    Priority:
    1. ARZULE_MODE env var (explicit)
    2. ARZULE_SELFHOSTED_ENDPOINT present → selfhosted
    3. ARZULE_API_KEY present → cloud
    4. Otherwise → local (graceful fallback)
    """
    # Explicit mode setting takes priority
    mode_str = os.environ.get("ARZULE_MODE", "").lower().strip()
    if mode_str:
        try:
            return ArzuleMode(mode_str)
        except ValueError:
            pass  # Invalid mode, continue with detection

    # Detect based on available credentials
    if os.environ.get("ARZULE_SELFHOSTED_ENDPOINT"):
        return ArzuleMode.SELFHOSTED
    elif os.environ.get("ARZULE_API_KEY"):
        return ArzuleMode.CLOUD
    else:
        return ArzuleMode.LOCAL


def get_default_output_path() -> str:
    """Get default output path for local mode traces."""
    return str(Path.home() / ".arzule" / "traces")


def create_mode_config(
    mode: Optional[str] = None,
    tenant_id: Optional[str] = None,
    project_id: Optional[str] = None,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    auth_type: Optional[str] = None,
    auth_value: Optional[str] = None,
    output_path: Optional[str] = None,
    destinations: Optional[list[dict[str, Any]]] = None,
    require_tls: bool = True,
    trace_collection_enabled: Optional[bool] = None,
) -> ModeConfig:
    """
    Create mode configuration from parameters and environment.

    Args:
        mode: SDK mode (cloud, local, selfhosted, multi). Auto-detected if not provided.
        tenant_id: Tenant identifier. Required for cloud mode (must be UUID).
        project_id: Project identifier. Required for cloud mode (must be UUID).
        api_key: API key for cloud mode.
        endpoint: Custom endpoint for selfhosted mode.
        auth_type: Authentication type (bearer, header, basic, none).
        auth_value: Authentication value/token.
        output_path: File path for local mode.
        destinations: List of destination configs for multi mode.
        require_tls: Require HTTPS for HTTP endpoints.
        trace_collection_enabled: Enable/disable trace collection.

    Returns:
        ModeConfig with resolved settings.
    """
    # Detect or parse mode
    detected_mode = ArzuleMode(mode) if mode else detect_mode()

    # Load trace collection setting from env if not provided
    if trace_collection_enabled is None:
        env_value = os.environ.get("ARZULE_TRACE_COLLECTION_ENABLED", "true").lower()
        trace_collection_enabled = env_value in {"1", "true", "yes", "y"}

    if detected_mode == ArzuleMode.CLOUD:
        return _create_cloud_config(
            tenant_id=tenant_id,
            project_id=project_id,
            api_key=api_key,
            endpoint=endpoint,
            require_tls=require_tls,
            trace_collection_enabled=trace_collection_enabled,
        )
    elif detected_mode == ArzuleMode.LOCAL:
        return _create_local_config(
            tenant_id=tenant_id,
            project_id=project_id,
            output_path=output_path,
            trace_collection_enabled=trace_collection_enabled,
        )
    elif detected_mode == ArzuleMode.SELFHOSTED:
        return _create_selfhosted_config(
            tenant_id=tenant_id,
            project_id=project_id,
            endpoint=endpoint,
            auth_type=auth_type,
            auth_value=auth_value,
            require_tls=require_tls,
            trace_collection_enabled=trace_collection_enabled,
        )
    elif detected_mode == ArzuleMode.MULTI:
        return _create_multi_config(
            destinations=destinations,
            trace_collection_enabled=trace_collection_enabled,
        )
    else:
        raise ValueError(f"Unknown mode: {detected_mode}")


def _create_cloud_config(
    tenant_id: Optional[str],
    project_id: Optional[str],
    api_key: Optional[str],
    endpoint: Optional[str],
    require_tls: bool,
    trace_collection_enabled: bool,
) -> ModeConfig:
    """Create config for Arzule cloud mode."""
    from .endpoints import get_ingest_url

    # Load from env if not provided
    api_key = api_key or os.environ.get("ARZULE_API_KEY")
    tenant_id = tenant_id or os.environ.get("ARZULE_TENANT_ID")
    project_id = project_id or os.environ.get("ARZULE_PROJECT_ID")
    endpoint = endpoint or os.environ.get("ARZULE_INGEST_URL") or get_ingest_url()

    # Validate required fields for cloud mode
    if not api_key:
        raise ValueError(
            "ARZULE_API_KEY is required for cloud mode. "
            "Set it as an environment variable, pass to init(), or use mode='local' for offline usage."
        )
    if not tenant_id or not project_id:
        raise ValueError(
            "ARZULE_TENANT_ID and ARZULE_PROJECT_ID are required for cloud mode. "
            "Set them as environment variables, pass to init(), or use mode='local' for offline usage."
        )

    # Validate UUID format for cloud mode
    _validate_uuid(tenant_id, "ARZULE_TENANT_ID")
    _validate_uuid(project_id, "ARZULE_PROJECT_ID")

    return ModeConfig(
        mode=ArzuleMode.CLOUD,
        tenant_id=tenant_id,
        project_id=project_id,
        api_key=api_key,
        endpoint=endpoint,
        auth_type=AuthType.BEARER,
        auth_value=api_key,
        require_tls=require_tls,
        trace_collection_enabled=trace_collection_enabled,
    )


def _create_local_config(
    tenant_id: Optional[str],
    project_id: Optional[str],
    output_path: Optional[str],
    trace_collection_enabled: bool,
) -> ModeConfig:
    """Create config for local file mode."""
    return ModeConfig(
        mode=ArzuleMode.LOCAL,
        tenant_id=tenant_id or os.environ.get("ARZULE_TENANT_ID") or "local",
        project_id=project_id or os.environ.get("ARZULE_PROJECT_ID") or "default",
        output_path=output_path or os.environ.get("ARZULE_OUTPUT_PATH") or get_default_output_path(),
        require_tls=False,  # Not applicable for local mode
        trace_collection_enabled=trace_collection_enabled,
    )


def _create_selfhosted_config(
    tenant_id: Optional[str],
    project_id: Optional[str],
    endpoint: Optional[str],
    auth_type: Optional[str],
    auth_value: Optional[str],
    require_tls: bool,
    trace_collection_enabled: bool,
) -> ModeConfig:
    """Create config for self-hosted backend mode."""
    # Load from env if not provided
    endpoint = endpoint or os.environ.get("ARZULE_SELFHOSTED_ENDPOINT")
    auth_type_str = auth_type or os.environ.get("ARZULE_AUTH_TYPE", "bearer")
    auth_value = auth_value or os.environ.get("ARZULE_AUTH_VALUE") or os.environ.get("ARZULE_API_KEY")

    if not endpoint:
        raise ValueError(
            "Endpoint is required for selfhosted mode. "
            "Set ARZULE_SELFHOSTED_ENDPOINT or pass endpoint to init()."
        )

    # Parse auth type
    try:
        parsed_auth_type = AuthType(auth_type_str.lower())
    except ValueError:
        raise ValueError(
            f"Invalid auth_type: {auth_type_str}. "
            f"Valid options: {', '.join(t.value for t in AuthType)}"
        )

    return ModeConfig(
        mode=ArzuleMode.SELFHOSTED,
        tenant_id=tenant_id or os.environ.get("ARZULE_TENANT_ID") or "selfhosted",
        project_id=project_id or os.environ.get("ARZULE_PROJECT_ID") or "default",
        endpoint=endpoint,
        auth_type=parsed_auth_type,
        auth_value=auth_value,
        require_tls=require_tls,
        trace_collection_enabled=trace_collection_enabled,
    )


def _create_multi_config(
    destinations: Optional[list[dict[str, Any]]],
    trace_collection_enabled: bool,
) -> ModeConfig:
    """Create config for multi-destination mode."""
    if not destinations:
        raise ValueError(
            "destinations list is required for multi mode. "
            "Example: destinations=[{'mode': 'cloud'}, {'mode': 'local'}]"
        )

    return ModeConfig(
        mode=ArzuleMode.MULTI,
        destinations=destinations,
        trace_collection_enabled=trace_collection_enabled,
    )


def _validate_uuid(value: str, field_name: str) -> None:
    """Validate that a value is a valid UUID (cloud mode only)."""
    import uuid

    try:
        uuid.UUID(value)
    except ValueError:
        display_value = f"{value[:50]}..." if len(value) > 50 else value
        raise ValueError(f"{field_name} must be a valid UUID, got: {display_value}")
