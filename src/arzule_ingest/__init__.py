"""Arzule Ingestion SDK - Capture multi-agent traces and send to Arzule."""

from __future__ import annotations

import atexit
import os
import sys
import threading
from typing import Any, Optional

from .run import ArzuleRun, current_run
from .config import ArzuleConfig
from .audit import AuditLogger, audit_log
from .endpoints import get_ingest_url, get_ingest_base_url, is_local_endpoint
from .mode import ArzuleMode, AuthType, detect_mode, create_mode_config

__version__ = "0.9.0"
__all__ = [
    "ArzuleRun",
    "current_run",
    "ArzuleConfig",
    "AuditLogger",
    "audit_log",
    "init",
    "new_run",
    "ensure_run",
    "shutdown",
    # New exports for self-hosting
    "ArzuleMode",
    "AuthType",
]

# Global state
_initialized = False
_global_sink: Optional[Any] = None  # TelemetrySink type
_global_run: Optional[ArzuleRun] = None
_config: Optional[dict] = None
_run_lock = threading.Lock()  # Thread-safe lock for new_run()
_run_started = False  # Track if a run has actually been entered

# Default ingest URL (use centralized endpoint)
DEFAULT_INGEST_URL = get_ingest_url()  # For backwards compatibility


def _check_crewai_available() -> bool:
    """Check if CrewAI is installed."""
    try:
        import crewai  # noqa: F401
        return True
    except ImportError:
        return False


def _check_langchain_available() -> bool:
    """Check if LangChain is installed."""
    try:
        import langchain_core  # noqa: F401
        return True
    except ImportError:
        try:
            import langchain  # noqa: F401
            return True
        except ImportError:
            return False


def _check_autogen_available() -> tuple[bool, str]:
    """Check if Microsoft AutoGen is installed and which version.
    
    Returns:
        Tuple of (is_available, version_type) where version_type is:
        - "v2" for AutoGen v0.7+ (autogen-core, autogen-agentchat)
        - "v0.2" for legacy AutoGen (pyautogen)
        - "" if not available
    """
    # Check for new AutoGen v0.7+ first
    try:
        import autogen_core  # noqa: F401
        import autogen_agentchat  # noqa: F401
        return True, "v2"
    except ImportError:
        pass
    
    # Check for legacy AutoGen v0.2
    try:
        import autogen  # noqa: F401
        # Make sure it's the old version, not a namespace package
        if hasattr(autogen, 'ConversableAgent'):
            return True, "v0.2"
    except ImportError:
        pass
    
    return False, ""


def _check_langgraph_available() -> bool:
    """Check if LangGraph is installed."""
    try:
        import langgraph  # noqa: F401
        return True
    except ImportError:
        return False


def init(
    # Mode selection (NEW)
    mode: Optional[str] = None,
    # Cloud mode (existing)
    api_key: Optional[str] = None,
    tenant_id: Optional[str] = None,
    project_id: Optional[str] = None,
    ingest_url: Optional[str] = None,
    # Self-hosted mode (NEW)
    endpoint: Optional[str] = None,
    auth_type: Optional[str] = None,
    auth_value: Optional[str] = None,
    # Local mode (NEW)
    output_path: Optional[str] = None,
    # Multi mode (NEW)
    destinations: Optional[list[dict[str, Any]]] = None,
    # Existing options
    auto_instrument: bool = True,
    require_tls: bool = True,
    trace_collection_enabled: Optional[bool] = None,
) -> dict:
    """
    Initialize Arzule with minimal configuration.

    This is the simplest way to get started. Call once at application startup:

        import arzule_ingest
        arzule_ingest.init()  # Auto-detects mode from environment

    Modes:
        - cloud (default if credentials present): Send traces to Arzule cloud
        - local (default if no credentials): Write traces to local files
        - selfhosted: Send traces to a custom HTTP backend
        - multi: Send traces to multiple destinations

    Examples:
        # Cloud mode (requires credentials)
        arzule_ingest.init()

        # Local mode (no credentials required)
        arzule_ingest.init(mode="local")
        arzule_ingest.init(mode="local", output_path="./traces/")

        # Self-hosted mode
        arzule_ingest.init(
            mode="selfhosted",
            endpoint="https://my-backend.com/ingest",
            auth_type="bearer",
            auth_value="my-token",
        )

        # Multi mode (multiple destinations)
        arzule_ingest.init(
            mode="multi",
            destinations=[
                {"mode": "cloud"},
                {"mode": "local", "output_path": "./backup/"},
            ]
        )

    Args:
        mode: SDK mode (cloud, local, selfhosted, multi). Auto-detected if not provided.
        api_key: API key for authentication. Defaults to ARZULE_API_KEY env var.
        tenant_id: Tenant ID. Required for cloud mode (must be UUID), optional for others.
        project_id: Project ID. Required for cloud mode (must be UUID), optional for others.
        ingest_url: Backend URL. Defaults to ARZULE_INGEST_URL or Arzule cloud.
        endpoint: Custom endpoint for selfhosted mode.
        auth_type: Authentication type for selfhosted mode (bearer, header, basic, none).
        auth_value: Authentication value/token for selfhosted mode.
        output_path: File path for local mode traces.
        destinations: List of destination configs for multi mode.
        auto_instrument: If True, automatically instruments CrewAI/LangChain/LangGraph/AutoGen.
        require_tls: If True, requires HTTPS (recommended for production).
        trace_collection_enabled: If False, traces are silently discarded (privacy opt-out).

    Returns:
        Config dict with mode, tenant_id, project_id for reference.

    Raises:
        ValueError: If required configuration is missing for the selected mode.
    """
    global _initialized, _global_sink, _global_run, _config

    if _initialized:
        return _config or {}

    # Create mode configuration (handles validation per mode)
    try:
        mode_config = create_mode_config(
            mode=mode,
            tenant_id=tenant_id,
            project_id=project_id,
            api_key=api_key,
            endpoint=endpoint or ingest_url,
            auth_type=auth_type,
            auth_value=auth_value,
            output_path=output_path,
            destinations=destinations,
            require_tls=require_tls,
            trace_collection_enabled=trace_collection_enabled,
        )
    except ValueError:
        # Re-raise validation errors
        raise

    # Create sink based on mode
    _global_sink = _create_sink_for_mode(mode_config)

    # Register cleanup on exit
    atexit.register(shutdown)

    # Auto-instrument frameworks
    _auto_instrument_frameworks(auto_instrument)

    _config = {
        "mode": mode_config.mode.value,
        "tenant_id": mode_config.tenant_id,
        "project_id": mode_config.project_id,
        "endpoint": mode_config.endpoint,
        "output_path": mode_config.output_path,
        "run_id": None,  # Will be set when run is created lazily
        "trace_collection_enabled": mode_config.trace_collection_enabled,
    }

    _initialized = True

    # Log initialization status
    if mode_config.trace_collection_enabled:
        if mode_config.mode == ArzuleMode.LOCAL:
            print(f"[arzule] Initialized in LOCAL mode (traces → {mode_config.output_path})", file=sys.stderr)
        elif mode_config.mode == ArzuleMode.SELFHOSTED:
            print(f"[arzule] Initialized in SELFHOSTED mode (endpoint: {mode_config.endpoint})", file=sys.stderr)
        elif mode_config.mode == ArzuleMode.MULTI:
            print(f"[arzule] Initialized in MULTI mode ({len(mode_config.destinations or [])} destinations)", file=sys.stderr)
        else:
            print(f"[arzule] Initialized in CLOUD mode (run will start on first crew kickoff)", file=sys.stderr)
    else:
        print(f"[arzule] Initialized with trace collection DISABLED (privacy opt-out)", file=sys.stderr)

    return _config


def _create_sink_for_mode(mode_config: Any) -> Any:
    """Create the appropriate sink based on mode configuration."""
    from .mode import ArzuleMode, ModeConfig

    if not isinstance(mode_config, ModeConfig):
        raise TypeError("mode_config must be a ModeConfig instance")

    if mode_config.mode == ArzuleMode.LOCAL:
        from .sinks.file_jsonl import JsonlFileSink
        from pathlib import Path

        output_path = mode_config.output_path or str(Path.home() / ".arzule" / "traces")
        # Create directory if it doesn't exist
        Path(output_path).mkdir(parents=True, exist_ok=True)
        # Use a timestamped file within the directory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = str(Path(output_path) / f"traces_{timestamp}.jsonl")

        return JsonlFileSink(
            path=file_path,
            compress=False,
            encrypt=False,
        )

    elif mode_config.mode == ArzuleMode.SELFHOSTED:
        from .sinks.http_batch import HttpBatchSink

        is_localhost = is_local_endpoint(mode_config.endpoint or "")
        return HttpBatchSink(
            endpoint_url=mode_config.endpoint,
            api_key=mode_config.auth_value,  # For backward compat
            auth_type=mode_config.auth_type,
            auth_value=mode_config.auth_value,
            require_tls=mode_config.require_tls and not is_localhost,
            trace_collection_enabled=mode_config.trace_collection_enabled,
        )

    elif mode_config.mode == ArzuleMode.MULTI:
        from .sinks.multi import MultiSink

        sinks = []
        for dest in mode_config.destinations or []:
            dest_mode = dest.get("mode", "cloud")
            dest_config = create_mode_config(
                mode=dest_mode,
                tenant_id=dest.get("tenant_id"),
                project_id=dest.get("project_id"),
                api_key=dest.get("api_key"),
                endpoint=dest.get("endpoint"),
                auth_type=dest.get("auth_type"),
                auth_value=dest.get("auth_value"),
                output_path=dest.get("output_path"),
                require_tls=dest.get("require_tls", True),
                trace_collection_enabled=mode_config.trace_collection_enabled,
            )
            sinks.append(_create_sink_for_mode(dest_config))

        return MultiSink(sinks)

    else:  # CLOUD mode (default)
        from .sinks.http_batch import HttpBatchSink

        endpoint = mode_config.endpoint or get_ingest_url()
        is_localhost = is_local_endpoint(endpoint)
        return HttpBatchSink(
            endpoint_url=endpoint,
            api_key=mode_config.api_key,
            require_tls=mode_config.require_tls and not is_localhost,
            trace_collection_enabled=mode_config.trace_collection_enabled,
        )


def _auto_instrument_frameworks(auto_instrument: bool) -> None:
    """Auto-instrument available frameworks."""
    if not auto_instrument:
        return

    # Auto-instrument CrewAI if available
    if _check_crewai_available():
        try:
            from .crewai.install import instrument_crewai
            instrument_crewai()
        except ImportError:
            print("[arzule] CrewAI not installed, skipping auto-instrumentation", file=sys.stderr)

    # Auto-instrument LangChain if available
    if _check_langchain_available():
        try:
            from .langchain.install import instrument_langchain
            instrument_langchain()
        except ImportError:
            print("[arzule] LangChain not installed, skipping auto-instrumentation", file=sys.stderr)

    # Auto-instrument AutoGen if available (detect version)
    autogen_available, autogen_version = _check_autogen_available()
    if autogen_available:
        if autogen_version == "v2":
            try:
                from .autogen_v2.install import instrument_autogen_v2
                instrument_autogen_v2()
            except ImportError as e:
                print(f"[arzule] AutoGen v0.7+ installed but integration failed: {e}", file=sys.stderr)
        elif autogen_version == "v0.2":
            try:
                from .autogen.install import instrument_autogen
                instrument_autogen()
            except ImportError as e:
                print(f"[arzule] AutoGen v0.2 installed but integration failed: {e}", file=sys.stderr)

    # Auto-instrument LangGraph if available
    if _check_langgraph_available():
        try:
            from .langgraph.install import instrument_langgraph
            instrument_langgraph()
        except ImportError:
            print("[arzule] LangGraph not installed, skipping auto-instrumentation", file=sys.stderr)


def ensure_run() -> Optional[str]:
    """
    Ensure a run exists, creating one if needed.

    This is called automatically when the first CrewAI crew kicks off.
    Unlike new_run(), this only creates a run if none exists yet.

    Returns:
        The run_id of the current (or newly created) run, or None if not initialized
        or if trace collection is disabled (privacy opt-out).
    """
    global _global_run, _config, _run_started

    if not _initialized or not _global_sink:
        return None

    # Privacy opt-out: don't create runs if trace collection is disabled
    if _config and not _config.get("trace_collection_enabled", True):
        return None
    
    with _run_lock:
        # If we already have an active run, return its ID
        if _global_run and _run_started:
            return _global_run.run_id
        
        # Create the first run
        tenant_id = _config.get("tenant_id") if _config else None
        project_id = _config.get("project_id") if _config else None
        
        if not tenant_id or not project_id:
            return None
        
        _global_run = ArzuleRun(
            tenant_id=tenant_id,
            project_id=project_id,
            sink=_global_sink,
        )
        _global_run.__enter__()
        _run_started = True
        
        # Update config with run_id
        if _config:
            _config["run_id"] = _global_run.run_id
        
        print(f"[arzule] Run started: {_global_run.run_id}", file=sys.stderr)
        
        return _global_run.run_id


def new_run() -> Optional[str]:
    """
    Start a new run, closing the previous one if any.

    This is called when you want to explicitly start a fresh run,
    for example when running multiple crews in sequence.

    For the first crew kickoff, use ensure_run() instead to avoid
    creating unnecessary empty runs.

    Returns:
        The new run_id, or None if not initialized or if trace collection
        is disabled (privacy opt-out).
    """
    global _global_run, _config, _run_started

    if not _initialized or not _global_sink:
        return None

    # Privacy opt-out: don't create runs if trace collection is disabled
    if _config and not _config.get("trace_collection_enabled", True):
        return None
    
    with _run_lock:
        # Close the previous run if any
        if _global_run and _run_started:
            try:
                _global_run.__exit__(None, None, None)
            except Exception:
                pass
        
        # CRITICAL: Force clear the sink buffer before starting new run
        # This prevents mixing events from different runs if flush failed
        if _global_sink and hasattr(_global_sink, 'clear_buffer'):
            cleared = _global_sink.clear_buffer()
            if cleared > 0:
                print(
                    f"[arzule] Warning: Cleared {cleared} unflushed events from previous run",
                    file=sys.stderr,
                )
        
        # CRITICAL: Clear handler caches to prevent stale run_id being used
        # by background threads that don't have the ContextVar set
        try:
            from .crewai.listener import clear_listener_cache
            clear_listener_cache()
        except ImportError:
            pass
        try:
            from .langchain.install import clear_handler_cache
            clear_handler_cache()
        except ImportError:
            pass
        
        # Create a new run with the same config
        tenant_id = _config.get("tenant_id") if _config else None
        project_id = _config.get("project_id") if _config else None
        
        if not tenant_id or not project_id:
            return None
        
        _global_run = ArzuleRun(
            tenant_id=tenant_id,
            project_id=project_id,
            sink=_global_sink,
        )
        _global_run.__enter__()
        _run_started = True
        
        # Update config with new run_id
        if _config:
            _config["run_id"] = _global_run.run_id
        
        return _global_run.run_id


def shutdown() -> None:
    """
    Shutdown Arzule and flush any pending events.

    This is called automatically on process exit, but can be called manually
    if you need to ensure events are flushed before continuing.
    """
    global _initialized, _global_sink, _global_run, _run_started

    if not _initialized:
        return

    # CRITICAL: Use lock and set _initialized=False FIRST to prevent race condition
    # where background threads call ensure_run() after run is closed but before
    # _initialized is set to False, which would create a ghost run
    with _run_lock:
        if not _initialized:  # Double-check under lock
            return
        
        _initialized = False  # Prevent any new runs from being created
        
        if _global_run and _run_started:
            try:
                _global_run.__exit__(None, None, None)
            except Exception:
                pass
            _global_run = None
            _run_started = False

    # Close sink outside the lock (can take time to flush)
    if _global_sink:
        try:
            _global_sink.close()
        except Exception:
            pass
        _global_sink = None


# Optional: Claude Agent SDK integration
try:
    from .agent_sdk import TracedClaudeClient
    __all__.append("TracedClaudeClient")
except ImportError:
    pass  # claude-agent-sdk not installed
