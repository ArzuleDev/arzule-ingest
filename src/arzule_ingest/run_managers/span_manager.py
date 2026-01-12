"""Span stack management for parent/child span relationships."""

import threading
from typing import Optional


class SpanManager:
    """
    Manages span stack and parent/child relationships.

    Used for sequential execution where spans form a tree via stack-based parent tracking.
    Supports flow -> method -> crew hierarchy for multi-crew orchestration.

    Now supports concurrent agent execution by maintaining per-agent span stacks
    keyed by agent_instance_id. This allows multiple agents to execute in parallel
    while maintaining correct parent-child span relationships for each agent.
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

        # Per-agent span stacks for concurrent execution
        # Key: agent_instance_id -> list of span_ids
        self._agent_span_stacks: dict[str, list[str]] = {}
        self._agent_span_lock = threading.Lock()

        # Map agent_instance_id to its root span (for agent-level hierarchy)
        self._agent_root_spans: dict[str, str] = {}
    
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

    # =========================================================================
    # Agent Instance Span Management (for concurrent agent execution)
    # =========================================================================

    def start_agent_span(self, agent_instance_id: str, span_id: str) -> None:
        """
        Start tracking spans for an agent instance.

        This sets the root span for the agent and initializes its span stack.

        Args:
            agent_instance_id: The unique instance identifier for the agent
            span_id: The root span ID for this agent's execution
        """
        with self._agent_span_lock:
            self._agent_root_spans[agent_instance_id] = span_id
            self._agent_span_stacks[agent_instance_id] = [span_id]

    def end_agent_span(self, agent_instance_id: str) -> Optional[str]:
        """
        End tracking spans for an agent instance.

        Cleans up the agent's span stack and root span tracking.

        Args:
            agent_instance_id: The unique instance identifier for the agent

        Returns:
            The root span ID that was tracked for this agent, or None
        """
        with self._agent_span_lock:
            self._agent_span_stacks.pop(agent_instance_id, None)
            return self._agent_root_spans.pop(agent_instance_id, None)

    def push_agent_span(self, agent_instance_id: str, span_id: str) -> None:
        """
        Push a span onto an agent's span stack.

        Args:
            agent_instance_id: The unique instance identifier for the agent
            span_id: The span ID to push
        """
        with self._agent_span_lock:
            if agent_instance_id not in self._agent_span_stacks:
                self._agent_span_stacks[agent_instance_id] = []
            self._agent_span_stacks[agent_instance_id].append(span_id)

    def pop_agent_span(self, agent_instance_id: str) -> Optional[str]:
        """
        Pop a span from an agent's span stack.

        Args:
            agent_instance_id: The unique instance identifier for the agent

        Returns:
            The popped span ID, or None if stack is empty
        """
        with self._agent_span_lock:
            stack = self._agent_span_stacks.get(agent_instance_id, [])
            if stack:
                return stack.pop()
            return None

    def get_agent_parent_span(self, agent_instance_id: str) -> Optional[str]:
        """
        Get the current parent span for an agent instance.

        Args:
            agent_instance_id: The unique instance identifier for the agent

        Returns:
            The span ID to use as parent for new spans within this agent's context,
            or falls back to the global parent span hierarchy if agent not tracked
        """
        with self._agent_span_lock:
            stack = self._agent_span_stacks.get(agent_instance_id, [])
            if stack:
                return stack[-1]
            # Fall back to agent root span if stack is empty
            root = self._agent_root_spans.get(agent_instance_id)
            if root:
                return root
        # Fall back to global hierarchy
        return self.current_parent_span_id()

    def get_agent_root_span(self, agent_instance_id: str) -> Optional[str]:
        """
        Get the root span for an agent instance.

        Args:
            agent_instance_id: The unique instance identifier for the agent

        Returns:
            The root span ID for this agent, or None if not tracked
        """
        with self._agent_span_lock:
            return self._agent_root_spans.get(agent_instance_id)

    def has_agent_span(self, agent_instance_id: str) -> bool:
        """
        Check if an agent instance is being tracked.

        Args:
            agent_instance_id: The unique instance identifier for the agent

        Returns:
            True if the agent has an active span context
        """
        with self._agent_span_lock:
            return agent_instance_id in self._agent_span_stacks

    def get_active_agent_instance_ids(self) -> list[str]:
        """
        Get all currently active agent instance IDs.

        Returns:
            List of agent_instance_ids that have active span contexts
        """
        with self._agent_span_lock:
            return list(self._agent_span_stacks.keys())

