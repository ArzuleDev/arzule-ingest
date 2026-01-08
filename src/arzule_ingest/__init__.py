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

__version__ = "0.7.9"
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
]

# Global state
_initialized = False
_global_sink: Optional[Any] = None  # TelemetrySink type
_global_run: Optional[ArzuleRun] = None
_config: Optional[dict] = None
_run_lock = threading.Lock()  # Thread-safe lock for new_run()
_run_started = False  # Track if a run has actually been entered

# Default ingest URL
DEFAULT_INGEST_URL = "https://uuczh0e8g5.execute-api.us-east-1.amazonaws.com/ingest"


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
    api_key: Optional[str] = None,
    tenant_id: Optional[str] = None,
    project_id: Optional[str] = None,
    ingest_url: Optional[str] = None,
    auto_instrument: bool = True,
    require_tls: bool = True,
) -> dict:
    """
    Initialize Arzule with minimal configuration.

    This is the simplest way to get started. Call once at application startup:

        import arzule_ingest
        arzule_ingest.init()

    Args:
        api_key: API key for authentication. Defaults to ARZULE_API_KEY env var.
        tenant_id: Tenant ID. Defaults to ARZULE_TENANT_ID env var.
        project_id: Project ID. Defaults to ARZULE_PROJECT_ID env var.
        ingest_url: Backend URL. Defaults to ARZULE_INGEST_URL or Arzule cloud.
        auto_instrument: If True, automatically instruments CrewAI/LangChain/LangGraph/AutoGen (if installed).
        require_tls: If True, requires HTTPS (recommended for production).

    Returns:
        Config dict with tenant_id, project_id for reference.

    Raises:
        ValueError: If required configuration is missing.
    """
    global _initialized, _global_sink, _global_run, _config

    if _initialized:
        return _config or {}

    # Load from env if not provided
    api_key = api_key or os.environ.get("ARZULE_API_KEY")
    tenant_id = tenant_id or os.environ.get("ARZULE_TENANT_ID")
    project_id = project_id or os.environ.get("ARZULE_PROJECT_ID")
    ingest_url = ingest_url or os.environ.get("ARZULE_INGEST_URL", DEFAULT_INGEST_URL)

    if not api_key:
        raise ValueError(
            "ARZULE_API_KEY is required. Set it as an environment variable or pass to init()."
        )

    if not tenant_id or not project_id:
        raise ValueError(
            "ARZULE_TENANT_ID and ARZULE_PROJECT_ID are required. "
            "Set them as environment variables or pass to init()."
        )

    # Validate that tenant_id and project_id are valid UUIDs
    import uuid
    try:
        uuid.UUID(tenant_id)
    except ValueError:
        raise ValueError(
            f"ARZULE_TENANT_ID must be a valid UUID, got: {tenant_id[:50]}..."
            if len(tenant_id) > 50 else f"ARZULE_TENANT_ID must be a valid UUID, got: {tenant_id}"
        )
    try:
        uuid.UUID(project_id)
    except ValueError:
        raise ValueError(
            f"ARZULE_PROJECT_ID must be a valid UUID, got: {project_id[:50]}..."
            if len(project_id) > 50 else f"ARZULE_PROJECT_ID must be a valid UUID, got: {project_id}"
        )

    # Create HTTP sink
    from .sinks.http_batch import HttpBatchSink

    # Allow HTTP for localhost in development
    is_localhost = "localhost" in ingest_url or "127.0.0.1" in ingest_url
    _global_sink = HttpBatchSink(
        endpoint_url=ingest_url,
        api_key=api_key,
        require_tls=require_tls and not is_localhost,
    )

    # NOTE: Don't create the run yet - defer until first crew kicks off
    # This prevents creating an empty "ghost" run when init() is called
    # before any crew execution. The run will be created by ensure_run()
    # which is called from _handle_crew_start.

    # Register cleanup on exit
    atexit.register(shutdown)

    # Auto-instrument CrewAI if available
    if auto_instrument and _check_crewai_available():
        try:
            from .crewai.install import instrument_crewai
            instrument_crewai()
        except ImportError:
            # CrewAI integration not available, that's ok
            print("[arzule] CrewAI not installed, skipping auto-instrumentation", file=sys.stderr)

    # Auto-instrument LangChain if available
    if auto_instrument and _check_langchain_available():
        try:
            from .langchain.install import instrument_langchain
            instrument_langchain()
        except ImportError:
            # LangChain integration not available, that's ok
            print("[arzule] LangChain not installed, skipping auto-instrumentation", file=sys.stderr)

    # Auto-instrument AutoGen if available (detect version)
    autogen_available, autogen_version = _check_autogen_available()
    if auto_instrument and autogen_available:
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
    if auto_instrument and _check_langgraph_available():
        try:
            from .langgraph.install import instrument_langgraph
            instrument_langgraph()
        except ImportError:
            # LangGraph integration not available, that's ok
            print("[arzule] LangGraph not installed, skipping auto-instrumentation", file=sys.stderr)

    _config = {
        "tenant_id": tenant_id,
        "project_id": project_id,
        "ingest_url": ingest_url,
        "run_id": None,  # Will be set when run is created lazily
    }

    _initialized = True

    print(f"[arzule] Initialized (run will start on first crew kickoff)", file=sys.stderr)

    return _config


def ensure_run() -> Optional[str]:
    """
    Ensure a run exists, creating one if needed.
    
    This is called automatically when the first CrewAI crew kicks off.
    Unlike new_run(), this only creates a run if none exists yet.
    
    Returns:
        The run_id of the current (or newly created) run, or None if not initialized.
    """
    global _global_run, _config, _run_started
    
    if not _initialized or not _global_sink:
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
        The new run_id, or None if not initialized.
    """
    global _global_run, _config, _run_started
    
    if not _initialized or not _global_sink:
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

