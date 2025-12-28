"""Span stack management for parent/child span relationships."""

from typing import Optional


class SpanManager:
    """
    Manages span stack and parent/child relationships.
    
    Used for sequential execution where spans form a tree via stack-based parent tracking.
    """
    
    def __init__(self, root_span_id: Optional[str] = None, crew_span_id: Optional[str] = None):
        """
        Initialize span manager.
        
        Args:
            root_span_id: The root span ID for the run
            crew_span_id: Optional crew-level span ID (parent for concurrent tasks)
        """
        self._root_span_id = root_span_id
        self._crew_span_id = crew_span_id
        self._span_stack: list[str] = []
    
    def current_parent_span_id(self) -> Optional[str]:
        """
        Get the current parent span ID (top of stack).
        
        Returns:
            The span ID to use as parent for new spans, or None
        """
        if self._span_stack:
            return self._span_stack[-1]
        return self._crew_span_id or self._root_span_id
    
    def push_span(self, span_id: str) -> None:
        """
        Push a span onto the stack (making it the new parent).
        
        Args:
            span_id: The span ID to push
        """
        self._span_stack.append(span_id)
    
    def pop_span(self) -> Optional[str]:
        """
        Pop a span from the stack.
        
        Returns:
            The popped span ID, or None if stack is empty
        """
        if self._span_stack:
            return self._span_stack.pop()
        return None
    
    def set_crew_span(self, span_id: str) -> None:
        """
        Set the crew-level span ID.
        
        Args:
            span_id: The crew span ID
        """
        self._crew_span_id = span_id
    
    def get_crew_span(self) -> Optional[str]:
        """
        Get the crew-level span ID.
        
        Returns:
            The crew span ID, or root span ID if crew span not set
        """
        return self._crew_span_id or self._root_span_id

