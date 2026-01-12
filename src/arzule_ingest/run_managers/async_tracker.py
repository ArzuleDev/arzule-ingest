"""Async context tracking for concurrent/async task execution."""

import threading
import time
import uuid
from typing import Any, Optional


class AsyncTracker:
    """
    Tracks async correlation IDs and pending async tasks.
    
    Provides async_id correlation for concurrent tasks and tracks pending
    async operations that need to complete before run ends.
    """
    
    def __init__(self):
        """Initialize async tracker."""
        # Async context tracking for concurrent/async task execution
        # Maps task_key -> {"async_id": str, "started_at": str, "parent_async_id": str | None}
        self._async_contexts: dict[str, dict[str, Any]] = {}
        self._async_contexts_lock = threading.Lock()
        
        # Pending async task tracking (for waiting before run.end)
        self._pending_async_tasks: set[str] = set()
        self._pending_async_lock = threading.Lock()
        self._async_complete_event = threading.Event()
    
    def start_async_context(self, task_key: str, parent_task_key: Optional[str] = None, now_fn: callable = None) -> str:
        """
        Create an async correlation ID for a concurrent task.
        
        Args:
            task_key: Unique identifier for the task
            parent_task_key: Optional parent task key for nested async
            now_fn: Function to get current timestamp (for testing)
        
        Returns:
            The async_id (UUID string) for this context
        """
        async_id = str(uuid.uuid4())
        parent_async_id = None
        
        if parent_task_key:
            parent_async_id = self.get_async_id(parent_task_key)
        
        # Get timestamp
        if now_fn:
            started_at = now_fn()
        else:
            from datetime import datetime, timezone
            started_at = datetime.now(timezone.utc).isoformat()
        
        with self._async_contexts_lock:
            self._async_contexts[task_key] = {
                "async_id": async_id,
                "started_at": started_at,
                "parent_async_id": parent_async_id,
            }
        
        return async_id
    
    def get_async_id(self, task_key: str) -> Optional[str]:
        """
        Get the async correlation ID for a task.
        
        Args:
            task_key: The task identifier
        
        Returns:
            The async_id if found, None otherwise
        """
        with self._async_contexts_lock:
            ctx = self._async_contexts.get(task_key)
            return ctx["async_id"] if ctx else None
    
    def end_async_context(self, task_key: str) -> Optional[str]:
        """
        End an async context and return its async_id.
        
        Args:
            task_key: The task identifier
        
        Returns:
            The async_id that was ended, or None if not found
        """
        with self._async_contexts_lock:
            ctx = self._async_contexts.pop(task_key, None)
            return ctx["async_id"] if ctx else None
    
    def get_async_context_info(self, task_key: str) -> Optional[dict[str, Any]]:
        """
        Get full async context info for a task.
        
        Args:
            task_key: The task identifier
        
        Returns:
            Dict with async_id, started_at, parent_async_id or None
        """
        with self._async_contexts_lock:
            return self._async_contexts.get(task_key)
    
    def has_async_contexts(self) -> bool:
        """
        Check if there are any active async contexts.
        
        Returns:
            True if any async contexts exist
        """
        with self._async_contexts_lock:
            return len(self._async_contexts) > 0
    
    def register_async_task(self, task_key: str) -> None:
        """
        Register an async task as pending.
        
        Used to track tasks that need to complete before run ends.
        
        Args:
            task_key: The task identifier
        """
        from ..logger import log_async_task_registered
        with self._pending_async_lock:
            self._pending_async_tasks.add(task_key)
            # Clear the event since we have pending work
            self._async_complete_event.clear()
        log_async_task_registered("", task_key)  # Empty run_id for now
    
    def complete_async_task(self, task_key: str) -> None:
        """
        Mark an async task as complete.
        
        Args:
            task_key: The task identifier
        """
        from ..logger import log_async_task_completed
        with self._pending_async_lock:
            self._pending_async_tasks.discard(task_key)
            # If no more pending tasks, signal completion
            if not self._pending_async_tasks:
                self._async_complete_event.set()
        log_async_task_completed("", task_key)  # Empty run_id for now
    
    def has_pending_async_tasks(self) -> bool:
        """
        Check if there are any pending async tasks.
        
        Returns:
            True if any tasks are pending
        """
        with self._pending_async_lock:
            return len(self._pending_async_tasks) > 0
    
    def get_pending_async_tasks(self) -> set[str]:
        """
        Get the set of pending async task keys.
        
        Returns:
            Set of task keys that are still pending
        """
        with self._pending_async_lock:
            return self._pending_async_tasks.copy()
    
    def wait_for_async_tasks(self, timeout: float = 30.0) -> bool:
        """
        Wait for all pending async tasks to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
        
        Returns:
            True if all tasks completed, False if timeout
        """
        with self._pending_async_lock:
            if not self._pending_async_tasks:
                return True
        
        # Wait for completion event or timeout
        completed = self._async_complete_event.wait(timeout=timeout)
        
        return completed

