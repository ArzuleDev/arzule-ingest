"""Thread-local agent context management."""

import threading
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


class AgentContext:
    """
    Manages thread-local agent context for LLM/tool event attribution.
    
    Each thread tracks its own active agent independently, allowing
    concurrent agent execution.
    """
    
    def __init__(self, run_id: str):
        """
        Initialize agent context.
        
        Args:
            run_id: The run ID for this context
        """
        self._run_id = run_id
    
    def set_current_agent(self, agent_info: dict[str, Any]) -> None:
        """
        Set the currently active agent for this thread.
        
        Args:
            agent_info: Dict with agent details (id, role, etc.)
        """
        _current_agent_local.agent = agent_info
        _current_agent_local.run_id = self._run_id
    
    def clear_current_agent(self) -> None:
        """Clear the thread-local agent context for this thread."""
        clear_current_agent_context()
    
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

