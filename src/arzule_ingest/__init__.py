"""Arzule Ingestion SDK - Capture multi-agent traces and send to Arzule."""

from __future__ import annotations

import atexit
import os
import sys
import threading
from typing import Optional

from .run import ArzuleRun, current_run
from .config import ArzuleConfig
from .audit import AuditLogger, audit_log

__version__ = "0.5.12"
__all__ = [
    "ArzuleRun",
    "current_run",
    "ArzuleConfig",
    "AuditLogger",
    "audit_log",
    "init",
    "new_run",
    "shutdown",
]

# Global state
_initialized = False
_global_sink: Optional["TelemetrySink"] = None
_global_run: Optional[ArzuleRun] = None
_config: Optional[dict] = None
_run_lock = threading.Lock()  # Thread-safe lock for new_run()

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


def _check_autogen_available() -> bool:
    """Check if Microsoft AutoGen is installed."""
    try:
        import autogen  # noqa: F401
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
        auto_instrument: If True, automatically instruments CrewAI/LangChain (if installed).
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

    # Create HTTP sink
    from .sinks.http_batch import HttpBatchSink

    # Allow HTTP for localhost in development
    is_localhost = "localhost" in ingest_url or "127.0.0.1" in ingest_url
    _global_sink = HttpBatchSink(
        endpoint_url=ingest_url,
        api_key=api_key,
        require_tls=require_tls and not is_localhost,
    )

    # Create a global run that auto-starts
    _global_run = ArzuleRun(
        tenant_id=tenant_id,
        project_id=project_id,
        sink=_global_sink,
    )
    _global_run.__enter__()

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

    # Auto-instrument AutoGen if available
    if auto_instrument and _check_autogen_available():
        try:
            from .autogen.install import instrument_autogen
            instrument_autogen()
        except ImportError:
            # AutoGen integration not available, that's ok
            print("[arzule] AutoGen not installed, skipping auto-instrumentation", file=sys.stderr)

    _config = {
        "tenant_id": tenant_id,
        "project_id": project_id,
        "ingest_url": ingest_url,
        "run_id": _global_run.run_id,
    }

    _initialized = True

    print(f"[arzule] Initialized. Run ID: {_global_run.run_id}", file=sys.stderr)

    return _config


def new_run() -> Optional[str]:
    """
    Start a new run, closing the previous one if any.
    
    This is called automatically when a new CrewAI crew kicks off, ensuring
    each crew execution gets its own run with sequence numbers starting at 1.
    
    Returns:
        The new run_id, or None if not initialized.
    """
    global _global_run, _config
    
    if not _initialized or not _global_sink:
        return None
    
    with _run_lock:
        # Close the previous run if any
        if _global_run:
            try:
                _global_run.__exit__(None, None, None)
            except Exception:
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
    global _initialized, _global_sink, _global_run

    if not _initialized:
        return

    if _global_run:
        try:
            _global_run.__exit__(None, None, None)
        except Exception:
            pass
        _global_run = None

    if _global_sink:
        try:
            _global_sink.close()
        except Exception:
            pass
        _global_sink = None

    _initialized = False

