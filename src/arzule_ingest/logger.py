"""Centralized logging for Arzule SDK.

Provides comprehensive logging for context fallback, event drops, and async timeouts.
Logs to stderr by default for visibility during development and debugging.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any, Optional

_logger: Optional[logging.Logger] = None
_logger_lock = threading.Lock()


def get_logger() -> logging.Logger:
    """Get or create the Arzule SDK logger.
    
    Returns a logger configured to output to stderr with a consistent format.
    Thread-safe singleton initialization.
    """
    global _logger
    if _logger is None:
        with _logger_lock:
            if _logger is None:
                _logger = logging.getLogger("arzule")
                _logger.setLevel(logging.DEBUG)
                handler = logging.StreamHandler(sys.stderr)
                handler.setFormatter(logging.Formatter(
                    "[arzule] %(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                ))
                # Only add handler if not already present
                if not _logger.handlers:
                    _logger.addHandler(handler)
    return _logger


def log_event_dropped(
    reason: str,
    event_class: str,
    extra: Optional[dict[str, Any]] = None
) -> None:
    """Log when an event is dropped due to context loss.
    
    Args:
        reason: Why the event was dropped (e.g., "no_active_run_and_fallback_failed")
        event_class: The class name of the dropped event
        extra: Additional context for debugging
    """
    extra_str = f" | {extra}" if extra else ""
    get_logger().warning(
        f"Event dropped: {reason} | event={event_class}{extra_str}"
    )


def log_context_fallback(run_id: str, source: str) -> None:
    """Log when falling back to global registry from ContextVar.
    
    This indicates the ContextVar failed (likely due to thread spawning)
    and the run was recovered from the global registry.
    
    Args:
        run_id: The run ID that was recovered
        source: Where the fallback occurred (e.g., "current_run", "listener")
    """
    get_logger().debug(
        f"Context fallback used | run_id={run_id} | source={source}"
    )


def log_async_timeout(run_id: str, pending_tasks: set[str]) -> None:
    """Log when async task wait times out.
    
    This indicates async tasks did not complete before the run.end timeout.
    
    Args:
        run_id: The run ID that timed out
        pending_tasks: Set of task keys that were still pending
    """
    get_logger().warning(
        f"Async timeout | run_id={run_id} | pending_tasks={pending_tasks}"
    )


def log_async_wait_start(run_id: str, pending_tasks: set[str]) -> None:
    """Log when starting to wait for async tasks.
    
    Args:
        run_id: The run ID
        pending_tasks: Set of task keys being waited on
    """
    get_logger().info(
        f"Waiting for async tasks | run_id={run_id} | pending_tasks={pending_tasks}"
    )


def log_run_registered(run_id: str) -> None:
    """Log when a run is registered in the global registry.
    
    Args:
        run_id: The run ID being registered
    """
    get_logger().debug(f"Run registered | run_id={run_id}")


def log_run_unregistered(run_id: str) -> None:
    """Log when a run is unregistered from the global registry.
    
    Args:
        run_id: The run ID being unregistered
    """
    get_logger().debug(f"Run unregistered | run_id={run_id}")


def log_async_task_registered(run_id: str, task_key: str) -> None:
    """Log when an async task is registered.
    
    Args:
        run_id: The run ID
        task_key: The task key being registered
    """
    get_logger().debug(
        f"Async task registered | run_id={run_id} | task_key={task_key}"
    )


def log_async_task_completed(run_id: str, task_key: str) -> None:
    """Log when an async task completes.
    
    Args:
        run_id: The run ID
        task_key: The task key that completed
    """
    get_logger().debug(
        f"Async task completed | run_id={run_id} | task_key={task_key}"
    )













