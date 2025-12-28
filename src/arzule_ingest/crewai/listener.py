"""CrewAI event listener for lifecycle events (CrewAI 1.7.x API).

Supports both sequential and concurrent (async_execution=True) task execution.
Uses per-task span tracking for proper parent-child relationships in concurrent mode.

Thread Safety:
- Caches run_id for fallback when ContextVar fails in spawned threads
- Uses global registry lookup when ContextVar returns None
- Comprehensive logging for dropped events and context fallback usage
"""

from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING, Any, Optional
import uuid

from ..ids import new_span_id
from ..logger import log_event_dropped
from ..run import current_run
from .handoff import (
    emit_handoff_ack,
    emit_handoff_complete,
    extract_handoff_key_from_text,
    is_delegation_tool,
    maybe_emit_handoff_proposed,
)
from .normalize import evt_from_crewai_event, evt_async_spawn, evt_async_join

if TYPE_CHECKING:
    from ..run import ArzuleRun

# Global singleton to prevent GC
_listener_instance: Optional["ArzuleCrewAIListener"] = None


def _get_task_key(event: Any) -> Optional[str]:
    """Extract a unique task key from an event."""
    task = getattr(event, "task", None)
    if task:
        # Try task.id first, then task.name, then generate from description hash
        task_id = getattr(task, "id", None)
        if task_id:
            return f"task:{task_id}"
        task_name = getattr(task, "name", None)
        if task_name:
            return f"task:{task_name}"
        # Fall back to description hash for uniqueness
        desc = getattr(task, "description", None)
        if desc:
            return f"task:desc:{hash(desc[:100])}"
    return None


def _get_agent_key(event: Any) -> Optional[str]:
    """Extract a unique agent key from an event."""
    agent = getattr(event, "agent", None)
    if agent:
        role = getattr(agent, "role", None)
        if role:
            return f"agent:{role}"
    return None


class ArzuleCrewAIListener:
    """
    Event listener for CrewAI lifecycle events.

    Listens to crew, agent, and task lifecycle events and emits trace events.
    Also handles handoff acknowledgment and completion detection.

    Supports concurrent task execution by maintaining per-task span trees,
    ensuring correct parent-child relationships even when multiple agents
    run in parallel with async_execution=True.

    NOTE: CrewAI event listeners can be garbage collected if not kept in memory.
    This class maintains a global singleton to prevent that.
    """

    def __init__(self) -> None:
        """Initialize and register the listener."""
        self._setup_complete = False
        # Track active concurrent tasks for debugging (thread-safe)
        self._active_task_keys: set[str] = set()
        self._active_task_keys_lock = threading.Lock()
        # Cache run_id for thread fallback when ContextVar fails
        self._cached_run_id: Optional[str] = None
        self._cached_run_id_lock = threading.Lock()

    # =========================================================================
    # Run Context Fallback (for spawned threads)
    # =========================================================================

    def _cache_run_id(self, run_id: str) -> None:
        """Cache the run_id for thread-safe fallback lookup.
        
        Called when we first see a valid run (crew start). This allows
        spawned threads to recover the run via global registry.
        """
        with self._cached_run_id_lock:
            self._cached_run_id = run_id

    def _clear_cached_run_id(self) -> None:
        """Clear the cached run_id when the crew ends."""
        with self._cached_run_id_lock:
            self._cached_run_id = None

    def _get_cached_run_id(self) -> Optional[str]:
        """Get the cached run_id (thread-safe)."""
        with self._cached_run_id_lock:
            return self._cached_run_id

    def _get_run_with_fallback(self, event_class: str) -> Optional["ArzuleRun"]:
        """Get run from ContextVar, falling back to cached run_id.
        
        This handles the case where CrewAI spawns threads that don't inherit
        the ContextVar. We use the cached run_id to look up the run in the
        global registry.
        
        Args:
            event_class: Name of the event class (for logging if dropped)
            
        Returns:
            The ArzuleRun instance, or None if not recoverable
        """
        # Try ContextVar first
        run = current_run()
        if run:
            return run
        
        # Fallback: try global registry with cached run_id
        cached_id = self._get_cached_run_id()
        if cached_id:
            run = current_run(run_id_hint=cached_id)
            if run:
                return run
        
        # Log the drop with context
        log_event_dropped(
            reason="no_active_run_and_fallback_failed",
            event_class=event_class,
            extra={"cached_run_id": cached_id}
        )
        return None

    def setup_listeners(self) -> None:
        """Set up event listeners on the CrewAI event bus."""
        if self._setup_complete:
            return

        try:
            from crewai.events.event_bus import crewai_event_bus
        except ImportError:
            import sys

            print("[arzule] CrewAI event bus not available", file=sys.stderr)
            return

        # Register for crew events
        self._register_crew_events(crewai_event_bus)

        # Register for agent events
        self._register_agent_events(crewai_event_bus)

        # Register for task events
        self._register_task_events(crewai_event_bus)

        # Register for tool events
        self._register_tool_events(crewai_event_bus)

        # Register for LLM events
        self._register_llm_events(crewai_event_bus)

        self._setup_complete = True

    def _register_crew_events(self, bus: Any) -> None:
        """Register crew lifecycle event handlers."""
        try:
            from crewai.events.types.crew_events import (
                CrewKickoffCompletedEvent,
                CrewKickoffFailedEvent,
                CrewKickoffStartedEvent,
            )

            @bus.on(CrewKickoffStartedEvent)
            def on_crew_start(source: Any, event: CrewKickoffStartedEvent) -> None:
                self._handle_crew_start(event)

            @bus.on(CrewKickoffCompletedEvent)
            def on_crew_complete(source: Any, event: CrewKickoffCompletedEvent) -> None:
                self._handle_crew_end(event, status="ok")

            @bus.on(CrewKickoffFailedEvent)
            def on_crew_failed(source: Any, event: CrewKickoffFailedEvent) -> None:
                self._handle_crew_end(event, status="error")

        except ImportError:
            pass

    def _register_agent_events(self, bus: Any) -> None:
        """Register agent lifecycle event handlers."""
        try:
            from crewai.events.types.agent_events import (
                AgentExecutionCompletedEvent,
                AgentExecutionErrorEvent,
                AgentExecutionStartedEvent,
            )

            @bus.on(AgentExecutionStartedEvent)
            def on_agent_start(source: Any, event: AgentExecutionStartedEvent) -> None:
                self._handle_event(event)

            @bus.on(AgentExecutionCompletedEvent)
            def on_agent_complete(source: Any, event: AgentExecutionCompletedEvent) -> None:
                self._handle_event(event)

            @bus.on(AgentExecutionErrorEvent)
            def on_agent_error(source: Any, event: AgentExecutionErrorEvent) -> None:
                self._handle_event(event)

        except ImportError:
            pass

    def _register_task_events(self, bus: Any) -> None:
        """Register task lifecycle event handlers."""
        try:
            from crewai.events.types.task_events import (
                TaskCompletedEvent,
                TaskFailedEvent,
                TaskStartedEvent,
            )

            @bus.on(TaskStartedEvent)
            def on_task_start(source: Any, event: TaskStartedEvent) -> None:
                self._handle_task_start(event)
                self._check_handoff_ack(event)

            @bus.on(TaskCompletedEvent)
            def on_task_complete(source: Any, event: TaskCompletedEvent) -> None:
                self._handle_task_end(event, status="ok")
                self._check_handoff_complete(event, status="ok")

            @bus.on(TaskFailedEvent)
            def on_task_failed(source: Any, event: TaskFailedEvent) -> None:
                self._handle_task_end(event, status="error")
                self._check_handoff_complete(event, status="error")

        except ImportError:
            pass

    def _register_tool_events(self, bus: Any) -> None:
        """Register tool usage event handlers."""
        try:
            from crewai.events.types.tool_usage_events import (
                ToolUsageFinishedEvent,
                ToolUsageStartedEvent,
                ToolUsageErrorEvent,
            )

            @bus.on(ToolUsageStartedEvent)
            def on_tool_start(source: Any, event: ToolUsageStartedEvent) -> None:
                self._handle_event(event)
                self._check_handoff_proposed(event)

            @bus.on(ToolUsageFinishedEvent)
            def on_tool_end(source: Any, event: ToolUsageFinishedEvent) -> None:
                self._handle_event(event)

            @bus.on(ToolUsageErrorEvent)
            def on_tool_error(source: Any, event: ToolUsageErrorEvent) -> None:
                self._handle_event(event)

        except ImportError:
            pass

    def _register_llm_events(self, bus: Any) -> None:
        """Register LLM call event handlers."""
        try:
            from crewai.events.types.llm_events import (
                LLMCallCompletedEvent,
                LLMCallFailedEvent,
                LLMCallStartedEvent,
            )

            @bus.on(LLMCallStartedEvent)
            def on_llm_start(source: Any, event: LLMCallStartedEvent) -> None:
                self._handle_event(event)

            @bus.on(LLMCallCompletedEvent)
            def on_llm_complete(source: Any, event: LLMCallCompletedEvent) -> None:
                self._handle_event(event)

            @bus.on(LLMCallFailedEvent)
            def on_llm_failed(source: Any, event: LLMCallFailedEvent) -> None:
                self._handle_event(event)

        except ImportError:
            pass

    # =========================================================================
    # Crew Lifecycle Handlers
    # =========================================================================

    def _handle_crew_start(self, event: Any) -> None:
        """Handle crew kickoff start - establish crew-level span."""
        run = current_run()
        if not run:
            return

        # Cache run_id for thread fallback (critical for async task support)
        self._cache_run_id(run.run_id)

        # Create crew-level span as parent for all task branches
        crew_span_id = new_span_id()
        run.set_crew_span(crew_span_id)

        trace_event = evt_from_crewai_event(run, event, parent_span_id=run._root_span_id)
        trace_event["span_id"] = crew_span_id
        run.emit(trace_event)

    def _handle_crew_end(self, event: Any, status: str) -> None:
        """Handle crew kickoff completion."""
        run = self._get_run_with_fallback(event.__class__.__name__)
        if not run:
            return

        trace_event = evt_from_crewai_event(
            run, event,
            parent_span_id=run._root_span_id,
        )
        trace_event["status"] = status
        run.emit(trace_event)

        # Clear cached run_id when crew ends
        self._clear_cached_run_id()

    # =========================================================================
    # Task Lifecycle Handlers (with per-task span tracking)
    # =========================================================================

    def _handle_task_start(self, event: Any) -> None:
        """
        Handle task start - create a new span tree branch for this task.

        This is critical for concurrent execution: each task gets its own
        span tree so that concurrent tasks don't corrupt each other's
        parent-child relationships.

        For async_execution=True tasks, also emits async.spawn event and
        starts an async context for proper correlation.
        """
        run = self._get_run_with_fallback(event.__class__.__name__)
        if not run:
            return

        task_key = _get_task_key(event)
        agent_key = _get_agent_key(event)

        # Check if this is an async execution task
        task = getattr(event, "task", None)
        is_async = task and getattr(task, "async_execution", False)

        if task_key:
            # Start a new span tree for this task, parented to crew span
            task_span_id = run.start_task_span(task_key, agent_key)
            parent_span_id = run.get_crew_span()

            # Thread-safe update of active task tracking
            with self._active_task_keys_lock:
                self._active_task_keys.add(task_key)
                concurrent_count = len(self._active_task_keys)

            # For async tasks, start async context and emit async.spawn
            async_id = None
            if is_async:
                async_id = run.start_async_context(task_key)
                # Register async task for run.end waiting
                run.register_async_task(task_key)
                
                # Extract agent info for the spawn event
                agent = getattr(event, "agent", None)
                agent_info = None
                if agent:
                    role = getattr(agent, "role", None) or getattr(agent, "name", "unknown")
                    agent_info = {"id": f"crewai:role:{role}", "role": role}
                
                # Emit async.spawn event
                spawn_event = evt_async_spawn(
                    run,
                    task_key=task_key,
                    async_id=async_id,
                    parent_span_id=parent_span_id,
                    task_description=getattr(task, "description", None),
                    agent_info=agent_info,
                )
                run.emit(spawn_event)

            # Emit task.start event with the new task span
            trace_event = evt_from_crewai_event(
                run, event,
                parent_span_id=parent_span_id,
                task_key=task_key,
            )
            trace_event["span_id"] = task_span_id
            trace_event["attrs_compact"]["task_key"] = task_key
            trace_event["attrs_compact"]["concurrent_tasks"] = concurrent_count
            
            # Add async context info if present
            if async_id:
                trace_event["attrs_compact"]["async_id"] = async_id
                trace_event["attrs_compact"]["async_execution"] = True
            
            run.emit(trace_event)
        else:
            # Fallback for events without task info
            trace_event = evt_from_crewai_event(run, event)
            run.emit(trace_event)

    def _handle_task_end(self, event: Any, status: str) -> None:
        """
        Handle task completion - close the task's span tree.
        
        For async_execution=True tasks, also emits async.join event and
        ends the async context.
        """
        run = self._get_run_with_fallback(event.__class__.__name__)
        if not run:
            return

        task_key = _get_task_key(event)

        if task_key:
            # Get task's root span before ending it
            task_span_id = run.get_task_root_span(task_key)
            parent_span_id = run.get_crew_span()

            # Check if this was an async task (has async context)
            async_id = run.get_async_id(task_key)

            # Emit task completion event
            trace_event = evt_from_crewai_event(
                run, event,
                parent_span_id=parent_span_id,
                task_key=task_key,
            )
            trace_event["status"] = status
            if task_span_id:
                trace_event["span_id"] = new_span_id()  # New span for the end event
                trace_event["attrs_compact"]["task_root_span"] = task_span_id
            trace_event["attrs_compact"]["task_key"] = task_key
            
            # Add async context info if present
            if async_id:
                trace_event["attrs_compact"]["async_id"] = async_id
            
            run.emit(trace_event)

            # For async tasks, emit async.join and end async context
            if async_id:
                # Extract agent info
                agent = getattr(event, "agent", None)
                agent_info = None
                if agent:
                    role = getattr(agent, "role", None) or getattr(agent, "name", "unknown")
                    agent_info = {"id": f"crewai:role:{role}", "role": role}
                
                # Get result summary if available
                result = getattr(event, "result", None) or getattr(event, "output", None)
                result_summary = None
                if result:
                    result_str = str(result)
                    result_summary = result_str[:100] + "..." if len(result_str) > 100 else result_str
                
                # Emit async.join event
                join_event = evt_async_join(
                    run,
                    task_key=task_key,
                    async_id=async_id,
                    parent_span_id=parent_span_id,
                    status=status,
                    result_summary=result_summary,
                    agent_info=agent_info,
                )
                run.emit(join_event)
                
                # End the async context
                run.end_async_context(task_key)
                # Mark async task as complete (signals run.end waiting)
                run.complete_async_task(task_key)

            # End the task's span tree
            run.end_task_span(task_key)

            # Thread-safe update of active task tracking
            with self._active_task_keys_lock:
                self._active_task_keys.discard(task_key)
        else:
            trace_event = evt_from_crewai_event(run, event)
            trace_event["status"] = status
            run.emit(trace_event)

    # =========================================================================
    # Generic Event Handler (for agent, tool, LLM events)
    # =========================================================================

    def _handle_event(self, event: Any) -> None:
        """
        Convert and emit a CrewAI event as a trace event.

        For concurrent execution, this determines the correct parent span
        by looking up the task associated with the event's agent.
        """
        run = self._get_run_with_fallback(event.__class__.__name__)
        if not run:
            return

        # Determine task context for proper parent span
        task_key = _get_task_key(event)
        agent_key = _get_agent_key(event)

        # Get the correct parent span based on task/agent context
        parent_span_id = run.get_task_parent_span(task_key=task_key, agent_key=agent_key)

        trace_event = evt_from_crewai_event(
            run, event,
            parent_span_id=parent_span_id,
            task_key=task_key,
        )

        # Add concurrency context to attrs for debugging
        if task_key:
            trace_event["attrs_compact"]["task_key"] = task_key
        if run.has_concurrent_tasks():
            trace_event["attrs_compact"]["concurrent_mode"] = True

        run.emit(trace_event)

    def _check_handoff_proposed(self, event: Any) -> None:
        """Check if this tool call is a delegation and emit handoff.proposed."""
        run = self._get_run_with_fallback(event.__class__.__name__)
        if not run:
            return

        tool_name = getattr(event, "tool_name", None)
        if not is_delegation_tool(tool_name):
            return

        # This is a delegation tool - emit handoff.proposed
        # Create a mock context object with the event attributes
        class _Context:
            pass

        context = _Context()
        context.tool_name = tool_name
        context.agent = getattr(event, "agent", None)

        # Extract tool_input - may be in different attributes
        tool_input = getattr(event, "tool_input", None)
        if tool_input is None:
            tool_input = getattr(event, "arguments", None)
        if tool_input is None:
            tool_input = {}

        # Ensure tool_input is a dict
        if not isinstance(tool_input, dict):
            tool_input = {"raw": str(tool_input)}

        # Generate handoff key and store in arzule namespace
        handoff_key = str(uuid.uuid4())
        if "arzule" not in tool_input:
            tool_input["arzule"] = {}
        tool_input["arzule"]["handoff_key"] = handoff_key

        context.tool_input = tool_input

        # Store pending handoff for correlation
        agent = getattr(event, "agent", None)
        to_coworker = tool_input.get("coworker") or tool_input.get("to") or tool_input.get("agent")
        run._handoff_pending[handoff_key] = {
            "from_role": getattr(agent, "role", None) if agent else None,
            "to_coworker": to_coworker,
            "proposed_at": run.now(),
        }

        # Emit handoff.proposed with a new span_id
        maybe_emit_handoff_proposed(run, context, span_id=new_span_id())

    def _check_handoff_ack(self, event: Any) -> None:
        """
        Check if this task start is acknowledging a handoff.
        
        IMPORTANT: Only matches via exact handoff_key extracted from task description.
        Fuzzy role-based matching has been removed to prevent ambiguous correlation
        in concurrent/async execution modes.
        """
        run = self._get_run_with_fallback(event.__class__.__name__)
        if not run:
            return

        # Extract task description to look for handoff marker
        task = getattr(event, "task", None)
        if not task:
            return

        description = getattr(task, "description", None)
        handoff_key = extract_handoff_key_from_text(description)

        # NOTE: Fuzzy role-based matching removed - require exact handoff_key only
        # This prevents ambiguous correlation in async execution mode where multiple
        # handoffs may target agents with similar roles simultaneously.

        if handoff_key and handoff_key in run._handoff_pending:
            agent = getattr(event, "agent", None)
            agent_role = getattr(agent, "role", None) if agent else None
            task_id = getattr(task, "id", None) or getattr(task, "name", None)

            emit_handoff_ack(
                run=run,
                handoff_key=handoff_key,
                task_id=task_id,
                agent_role=agent_role,
            )

            # Mark as acked for correlation
            run._handoff_pending[handoff_key]["acked"] = True

    def _check_handoff_complete(self, event: Any, status: str) -> None:
        """Check if this task completion completes a handoff."""
        run = self._get_run_with_fallback(event.__class__.__name__)
        if not run:
            return

        task = getattr(event, "task", None)
        if not task:
            return

        description = getattr(task, "description", None)
        handoff_key = extract_handoff_key_from_text(description)

        # Only proceed if explicit handoff_key found in description
        # Fuzzy role-based matching is disabled (see _find_acked_handoff_for_role)
        if handoff_key and handoff_key in run._handoff_pending:
            agent = getattr(event, "agent", None)
            agent_role = getattr(agent, "role", None) if agent else None
            task_id = getattr(task, "id", None) or getattr(task, "name", None)

            # Get result summary if available
            result = getattr(event, "result", None) or getattr(event, "output", None)
            result_summary = None
            if result:
                result_str = str(result)
                result_summary = result_str[:100] + "..." if len(result_str) > 100 else result_str

            emit_handoff_complete(
                run=run,
                handoff_key=handoff_key,
                task_id=task_id,
                agent_role=agent_role,
                status=status,
                result_summary=result_summary,
            )

    def _find_pending_handoff_for_role(self, run: Any, agent_role: str) -> Optional[str]:
        """
        Find a pending handoff that targets the given agent role.
        
        DEPRECATED: This method now returns None to enforce exact handoff_key matching.
        Fuzzy role-based matching caused false correlations in async execution mode
        where multiple concurrent handoffs could target agents with similar roles.
        
        The handoff_key is now the only correlation mechanism, injected into
        task descriptions via [arzule_handoff:UUID] marker.
        """
        # Fuzzy matching disabled - return None to force exact handoff_key matching
        # See _check_handoff_ack for the exact matching logic
        return None

    def _find_acked_handoff_for_role(self, run: Any, agent_role: str) -> Optional[str]:
        """
        Find an acked handoff that was targeting the given agent role.
        
        DEPRECATED: This method now returns None to enforce exact handoff_key matching.
        See _find_pending_handoff_for_role for rationale.
        """
        # Fuzzy matching disabled - return None to force exact handoff_key matching
        return None


def get_listener() -> ArzuleCrewAIListener:
    """Get or create the global listener singleton."""
    global _listener_instance
    if _listener_instance is None:
        _listener_instance = ArzuleCrewAIListener()
    return _listener_instance
