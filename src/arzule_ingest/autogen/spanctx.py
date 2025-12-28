"""Span context management for AutoGen conversations."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional

from ..ids import new_span_id

if TYPE_CHECKING:
    from ..run import ArzuleRun


# Context variable for tracking conversation spans
# Maps conversation_id -> span_id
_conversation_spans: ContextVar[dict[str, str]] = ContextVar(
    "_autogen_conversation_spans", default={}
)

# Context variable for the current active conversation
_current_conversation: ContextVar[Optional[str]] = ContextVar(
    "_autogen_current_conversation", default=None
)


def get_conversation_span_id(conversation_id: str) -> Optional[str]:
    """Get the span ID for a conversation."""
    spans = _conversation_spans.get()
    return spans.get(conversation_id)


def get_current_conversation_id() -> Optional[str]:
    """Get the current active conversation ID."""
    return _current_conversation.get()


def start_conversation_span(run: "ArzuleRun", conversation_id: str) -> str:
    """
    Start a new span for a conversation.

    Args:
        run: The active ArzuleRun
        conversation_id: Unique identifier for this conversation

    Returns:
        The new span ID
    """
    span_id = new_span_id()
    
    # Store in conversation spans map
    spans = _conversation_spans.get().copy()
    spans[conversation_id] = span_id
    _conversation_spans.set(spans)
    
    # Set as current conversation
    _current_conversation.set(conversation_id)
    
    # Push to run's span stack
    run.push_span(span_id)
    
    return span_id


def end_conversation_span(run: "ArzuleRun", conversation_id: str) -> Optional[str]:
    """
    End a conversation span.

    Args:
        run: The active ArzuleRun
        conversation_id: The conversation to end

    Returns:
        The ended span ID
    """
    spans = _conversation_spans.get().copy()
    span_id = spans.pop(conversation_id, None)
    _conversation_spans.set(spans)
    
    # Clear current conversation if it matches
    if _current_conversation.get() == conversation_id:
        _current_conversation.set(None)
    
    # Pop from run's span stack
    if span_id:
        run.pop_span()
    
    return span_id


def get_message_span_id() -> str:
    """Generate a new span ID for a message within a conversation."""
    return new_span_id()


