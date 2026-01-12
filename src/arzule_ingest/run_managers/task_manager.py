"""Task and concurrent execution tracking."""

import threading
from typing import Any, Optional

from ..ids import new_span_id


class TaskManager:
    """
    Manages per-task span tracking for concurrent execution.
    
    Each task gets its own span tree, allowing multiple tasks to execute
    concurrently without interfering with each other's span hierarchies.
    """
    
    def __init__(self, root_span_id: Optional[str] = None, crew_span_id: Optional[str] = None):
        """
        Initialize task manager.
        
        Args:
            root_span_id: The root span ID for fallback
            crew_span_id: The crew-level span ID for fallback
        """
        self._root_span_id = root_span_id
        self._crew_span_id = crew_span_id
        
        # Per-task span tracking
        # Maps task_key -> {"root_span_id": str, "current_span_id": str, "stack": list[str], "agent_key": str | None}
        self._task_spans: dict[str, dict[str, Any]] = {}
        self._task_spans_lock = threading.Lock()
        
        # Maps agent_key -> current task_key (for correlating agent events to tasks)
        self._agent_task_map: dict[str, str] = {}
    
    def start_task_span(self, task_key: str, agent_key: Optional[str] = None) -> str:
        """
        Start a new span tree for a task (used for concurrent execution).
        
        Args:
            task_key: Unique identifier for the task
            agent_key: Optional agent key to associate with this task
        
        Returns:
            The root span ID for this task
        """
        span_id = new_span_id()
        with self._task_spans_lock:
            self._task_spans[task_key] = {
                "root_span_id": span_id,
                "current_span_id": span_id,
                "stack": [span_id],
                "agent_key": agent_key,
            }
            if agent_key:
                self._agent_task_map[agent_key] = task_key
        return span_id
    
    def end_task_span(self, task_key: str) -> Optional[str]:
        """
        End a task's span tree.
        
        Args:
            task_key: The task identifier
        
        Returns:
            The root span ID that was ended, or None
        """
        with self._task_spans_lock:
            task_info = self._task_spans.pop(task_key, None)
            if task_info:
                agent_key = task_info.get("agent_key")
                if agent_key and self._agent_task_map.get(agent_key) == task_key:
                    self._agent_task_map.pop(agent_key, None)
                return task_info["root_span_id"]
        return None
    
    def get_task_parent_span(self, task_key: Optional[str] = None, agent_key: Optional[str] = None) -> Optional[str]:
        """
        Get the current parent span for a task.
        
        Args:
            task_key: The task identifier
            agent_key: Alternative - look up task by agent key
        
        Returns:
            The current parent span ID for the task, or crew/root span if not found
        """
        with self._task_spans_lock:
            # Try task_key first
            if task_key and task_key in self._task_spans:
                return self._task_spans[task_key]["current_span_id"]
            
            # Fall back to agent_key lookup
            if agent_key and agent_key in self._agent_task_map:
                task_key = self._agent_task_map[agent_key]
                if task_key in self._task_spans:
                    return self._task_spans[task_key]["current_span_id"]
        
        # Fall back to crew span or root span
        return self._crew_span_id or self._root_span_id
    
    def get_task_root_span(self, task_key: str) -> Optional[str]:
        """
        Get the root span ID for a task.
        
        Args:
            task_key: The task identifier
        
        Returns:
            The root span ID for this task, or None
        """
        with self._task_spans_lock:
            task_info = self._task_spans.get(task_key)
            return task_info["root_span_id"] if task_info else None
    
    def push_task_span(self, task_key: str, span_id: str) -> None:
        """
        Push a child span onto a task's span stack.
        
        Args:
            task_key: The task identifier
            span_id: The span ID to push
        """
        with self._task_spans_lock:
            if task_key in self._task_spans:
                self._task_spans[task_key]["stack"].append(span_id)
                self._task_spans[task_key]["current_span_id"] = span_id
    
    def pop_task_span(self, task_key: str) -> Optional[str]:
        """
        Pop a span from a task's span stack.
        
        Args:
            task_key: The task identifier
        
        Returns:
            The popped span ID, or None
        """
        with self._task_spans_lock:
            if task_key in self._task_spans:
                stack = self._task_spans[task_key]["stack"]
                if len(stack) > 1:  # Keep at least the root
                    popped = stack.pop()
                    self._task_spans[task_key]["current_span_id"] = stack[-1]
                    return popped
        return None
    
    def get_task_key_for_agent(self, agent_key: str) -> Optional[str]:
        """
        Get the current task key associated with an agent.
        
        Args:
            agent_key: The agent identifier
        
        Returns:
            The task key, or None
        """
        with self._task_spans_lock:
            return self._agent_task_map.get(agent_key)
    
    def associate_agent_with_task(self, agent_key: str, task_key: str) -> None:
        """
        Associate an agent with a task.
        
        Args:
            agent_key: The agent identifier
            task_key: The task identifier
        """
        with self._task_spans_lock:
            self._agent_task_map[agent_key] = task_key
    
    def has_concurrent_tasks(self) -> bool:
        """
        Check if there are multiple concurrent tasks active.
        
        Returns:
            True if 2+ tasks are active
        """
        with self._task_spans_lock:
            return len(self._task_spans) > 1

