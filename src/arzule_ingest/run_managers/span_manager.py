"""Span stack management for parent/child span relationships."""

from typing import Optional


class SpanManager:
    """
    Manages span stack and parent/child relationships.
    
    Used for sequential execution where spans form a tree via stack-based parent tracking.
    Supports flow -> method -> crew hierarchy for multi-crew orchestration.
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
        
        # Flow tracking (for multi-crew orchestration)
        self._flow_span_id: Optional[str] = None
        self._method_span_id: Optional[str] = None
    
    def current_parent_span_id(self) -> Optional[str]:
        """
        Get the current parent span ID (top of stack).
        
        Returns:
            The span ID to use as parent for new spans, or None
        """
        if self._span_stack:
            return self._span_stack[-1]
        return self._crew_span_id or self._method_span_id or self._flow_span_id or self._root_span_id
    
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
            The crew span ID, or method/flow/root span ID if crew span not set
        """
        return self._crew_span_id or self._method_span_id or self._flow_span_id or self._root_span_id
    
    def clear_crew_span(self) -> None:
        """Clear the crew-level span ID when a crew ends."""
        self._crew_span_id = None
    
    # =========================================================================
    # Flow Span Management (for multi-crew orchestration)
    # =========================================================================
    
    def set_flow_span(self, span_id: str) -> None:
        """
        Set the flow-level span ID.
        
        Args:
            span_id: The flow span ID
        """
        self._flow_span_id = span_id
    
    def get_flow_span(self) -> Optional[str]:
        """
        Get the flow-level span ID.
        
        Returns:
            The flow span ID, or None if not in a flow
        """
        return self._flow_span_id
    
    def clear_flow_span(self) -> None:
        """Clear the flow-level span ID when a flow ends."""
        self._flow_span_id = None
    
    def has_flow_context(self) -> bool:
        """
        Check if we are currently inside a flow.
        
        Returns:
            True if a flow span is set, False otherwise
        """
        return self._flow_span_id is not None
    
    # =========================================================================
    # Method Span Management (for flow method tracking)
    # =========================================================================
    
    def set_method_span(self, span_id: str) -> None:
        """
        Set the current flow method span ID.
        
        Args:
            span_id: The method span ID
        """
        self._method_span_id = span_id
    
    def get_method_span(self) -> Optional[str]:
        """
        Get the current flow method span ID.
        
        Returns:
            The method span ID, or flow span if not in a method
        """
        return self._method_span_id or self._flow_span_id
    
    def clear_method_span(self) -> None:
        """Clear the method-level span ID when a method ends."""
        self._method_span_id = None

