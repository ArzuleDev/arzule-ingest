"""Span context management for AutoGen v0.7+ tracing.

Manages span IDs and context for nested agent calls and operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import threading

from ..ids import new_span_id

if TYPE_CHECKING:
    from ..run import ArzuleRun

# Thread-local storage for agent spans
_agent_spans: dict[str, list[str]] = {}
_agent_spans_lock = threading.Lock()


def start_agent_span(run: "ArzuleRun", agent_name: str) -> str:
    """
    Start a new span for an agent invocation.
    
    Args:
        run: The active ArzuleRun
        agent_name: Name of the agent
        
    Returns:
        The new span ID
    """
    span_id = new_span_id()
    
    with _agent_spans_lock:
        run_key = f"{run.run_id}:{agent_name}"
        if run_key not in _agent_spans:
            _agent_spans[run_key] = []
        _agent_spans[run_key].append(span_id)
    
    return span_id


def end_agent_span(run: "ArzuleRun", agent_name: str) -> Optional[str]:
    """
    End the current span for an agent invocation.
    
    Args:
        run: The active ArzuleRun
        agent_name: Name of the agent
        
    Returns:
        The span ID that was ended, or None if no span was active
    """
    with _agent_spans_lock:
        run_key = f"{run.run_id}:{agent_name}"
        if run_key in _agent_spans and _agent_spans[run_key]:
            return _agent_spans[run_key].pop()
    
    return None


def get_current_span_id(run: "ArzuleRun") -> Optional[str]:
    """
    Get the current parent span ID from the run's span stack.
    
    Args:
        run: The active ArzuleRun
        
    Returns:
        The current parent span ID, or None if no span is active
    """
    # Use the run's span stack to get the current parent span ID
    return run.current_parent_span_id()


def clear_agent_spans(run: "ArzuleRun") -> None:
    """
    Clear all agent spans for a run.
    
    Args:
        run: The ArzuleRun to clear spans for
    """
    with _agent_spans_lock:
        keys_to_remove = [k for k in _agent_spans.keys() if k.startswith(f"{run.run_id}:")]
        for key in keys_to_remove:
            del _agent_spans[key]

