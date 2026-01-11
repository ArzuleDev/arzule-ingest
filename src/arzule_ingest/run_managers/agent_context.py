"""Thread-local agent context management with instance-aware tracking."""

import threading
import uuid
from typing import Any, Optional


# Thread-local storage for current agent context
_current_agent_local = threading.local()


def clear_current_agent_context() -> None:
    """
    Clear the thread-local agent context.

    This is a module-level function that can be called even when no run is available.
    Used to ensure agent context is always cleared when an agent ends, preventing
    stale context from persisting if run lookup fails.
    """
    _current_agent_local.agent = None
    _current_agent_local.run_id = None
    _current_agent_local.parent_agent_id = None


def _generate_instance_id() -> str:
    """Generate a unique instance identifier for agent tracking."""
    return uuid.uuid4().hex[:8]


class AgentContext:
    """
    Manages thread-local agent context for LLM/tool event attribution.

    Each thread tracks its own active agent independently, allowing
    concurrent agent execution. Now supports instance-aware tracking
    where each agent instance gets a unique ID regardless of role.

    Key features:
    - Instance-aware IDs: Each agent gets a unique instance_id
    - Parent tracking: Tracks which agent spawned this one (for delegation)
    - Agent stack: Supports nested agent contexts (agent spawns agent)
    """

    def __init__(self, run_id: str):
        """
        Initialize agent context.

        Args:
            run_id: The run ID for this context
        """
        self._run_id = run_id
        # Lock for thread-safe access to agent tracking dicts
        self._lock = threading.Lock()
        # Map from agent_instance_id to agent_info dict
        self._agent_instances: dict[str, dict[str, Any]] = {}
        # Stack of agent instance IDs for nested agent contexts (per-thread)
        # This is thread-local since different threads may have different agent stacks
        self._agent_stack_local = threading.local()

    def _get_agent_stack(self) -> list[str]:
        """Get the agent stack for the current thread."""
        if not hasattr(self._agent_stack_local, "stack"):
            self._agent_stack_local.stack = []
        return self._agent_stack_local.stack

    def set_current_agent(self, agent_info: dict[str, Any]) -> str:
        """
        Set the currently active agent for this thread.

        If the agent_info does not have an instance_id, one will be generated.
        The current agent (if any) becomes the parent of this new agent.

        Args:
            agent_info: Dict with agent details (id, role, instance_id, etc.)

        Returns:
            The instance_id of the agent (for later reference)
        """
        # Ensure agent has an instance_id
        instance_id = agent_info.get("instance_id")
        if not instance_id:
            instance_id = _generate_instance_id()
            agent_info["instance_id"] = instance_id
            # Update ID to be role-based for visualization swimlane consolidation
            role = agent_info.get("role", "unknown")
            agent_info["id"] = f"crewai:role:{role}"

        # Get the current agent (if any) to set as parent
        current_agent = self.get_current_agent()
        if current_agent and "parent_agent_id" not in agent_info:
            # Set parent to the current agent's full ID
            agent_info["parent_agent_id"] = current_agent.get("id")

        # Store the agent instance
        with self._lock:
            self._agent_instances[instance_id] = agent_info

        # Push onto the agent stack for this thread
        agent_stack = self._get_agent_stack()
        agent_stack.append(instance_id)

        # Update thread-local current agent
        _current_agent_local.agent = agent_info
        _current_agent_local.run_id = self._run_id
        _current_agent_local.parent_agent_id = agent_info.get("parent_agent_id")

        return instance_id

    def clear_current_agent(self) -> Optional[str]:
        """
        Clear the current agent and pop from the agent stack.

        If there's a parent agent on the stack, it becomes the new current agent.

        Returns:
            The instance_id of the cleared agent, or None if no agent was active
        """
        agent_stack = self._get_agent_stack()

        if not agent_stack:
            clear_current_agent_context()
            return None

        # Pop the current agent
        cleared_instance_id = agent_stack.pop()

        if agent_stack:
            # Restore the parent agent as current
            parent_instance_id = agent_stack[-1]
            with self._lock:
                parent_agent = self._agent_instances.get(parent_instance_id)
            if parent_agent:
                _current_agent_local.agent = parent_agent
                _current_agent_local.run_id = self._run_id
                _current_agent_local.parent_agent_id = parent_agent.get("parent_agent_id")
            else:
                clear_current_agent_context()
        else:
            clear_current_agent_context()

        return cleared_instance_id

    def get_current_agent(self) -> Optional[dict[str, Any]]:
        """
        Get the currently active agent for this thread.

        Only returns agent if it belongs to this run (prevents cross-run pollution).

        Returns:
            Agent info dict if set and belongs to this run, None otherwise
        """
        agent = getattr(_current_agent_local, "agent", None)
        stored_run_id = getattr(_current_agent_local, "run_id", None)

        # Only return agent if it belongs to this run
        if agent and stored_run_id == self._run_id:
            return agent
        return None

    def get_current_agent_instance_id(self) -> Optional[str]:
        """
        Get the instance_id of the currently active agent.

        Returns:
            The instance_id string or None if no agent is active
        """
        agent = self.get_current_agent()
        return agent.get("instance_id") if agent else None

    def get_parent_agent_id(self) -> Optional[str]:
        """
        Get the ID of the parent agent that spawned the current agent.

        Returns:
            The parent agent's full ID or None if no parent
        """
        return getattr(_current_agent_local, "parent_agent_id", None)

    def get_agent_by_instance_id(self, instance_id: str) -> Optional[dict[str, Any]]:
        """
        Get agent info by instance_id.

        Args:
            instance_id: The unique instance identifier

        Returns:
            Agent info dict or None if not found
        """
        with self._lock:
            return self._agent_instances.get(instance_id)

    def set_parent_agent(self, agent_info: dict[str, Any], parent_agent_id: str) -> None:
        """
        Explicitly set the parent agent for an agent.

        Args:
            agent_info: The agent info dict to update
            parent_agent_id: The full ID of the parent agent
        """
        agent_info["parent_agent_id"] = parent_agent_id
        instance_id = agent_info.get("instance_id")
        if instance_id:
            with self._lock:
                if instance_id in self._agent_instances:
                    self._agent_instances[instance_id]["parent_agent_id"] = parent_agent_id

