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
from ..run import clear_current_agent_context, current_run
from .handoff import (
    emit_handoff_ack,
    emit_handoff_complete,
    extract_handoff_key_from_text,
    is_delegation_tool,
    maybe_emit_handoff_proposed,
)
from .implicit_handoff import (
    detect_task_context_handoff,
    detect_sequential_handoff,
    emit_implicit_handoff_complete,
    cleanup_run_tracking,
)
from .normalize import (
    evt_from_crewai_event,
    evt_async_spawn,
    evt_async_join,
    evt_flow_start,
    evt_flow_end,
    evt_method_start,
    evt_method_end,
    extract_agent_info_from_event,
)

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

        # Register for flow events (for multi-crew orchestration)
        self._register_flow_events(crewai_event_bus)

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

        # Register for memory/knowledge/reasoning events (optional - newer CrewAI versions)
        self._register_memory_events(crewai_event_bus)

        self._setup_complete = True

    def _register_flow_events(self, bus: Any) -> None:
        """Register flow lifecycle event handlers for multi-crew orchestration."""
        try:
            from crewai.events.types.flow_events import (
                FlowStartedEvent,
                FlowFinishedEvent,
                MethodExecutionStartedEvent,
                MethodExecutionFinishedEvent,
                MethodExecutionFailedEvent,
            )

            @bus.on(FlowStartedEvent)
            def on_flow_start(source: Any, event: FlowStartedEvent) -> None:
                self._handle_flow_start(event)

            @bus.on(FlowFinishedEvent)
            def on_flow_end(source: Any, event: FlowFinishedEvent) -> None:
                self._handle_flow_end(event, status="ok")

            @bus.on(MethodExecutionStartedEvent)
            def on_method_start(source: Any, event: MethodExecutionStartedEvent) -> None:
                self._handle_method_start(event)

            @bus.on(MethodExecutionFinishedEvent)
            def on_method_end(source: Any, event: MethodExecutionFinishedEvent) -> None:
                self._handle_method_end(event, status="ok")

            @bus.on(MethodExecutionFailedEvent)
            def on_method_failed(source: Any, event: MethodExecutionFailedEvent) -> None:
                self._handle_method_end(event, status="error")

        except ImportError:
            # Flow events not available in this CrewAI version
            pass

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
        """Register agent lifecycle event handlers.
        
        Agent events use dedicated handlers to track the current agent in
        thread-local storage. This enables LLM and tool events (which often
        don't include agent context from CrewAI) to be attributed to the
        correct agent.
        """
        try:
            from crewai.events.types.agent_events import (
                AgentExecutionCompletedEvent,
                AgentExecutionErrorEvent,
                AgentExecutionStartedEvent,
            )

            @bus.on(AgentExecutionStartedEvent)
            def on_agent_start(source: Any, event: AgentExecutionStartedEvent) -> None:
                self._handle_agent_start(event)

            @bus.on(AgentExecutionCompletedEvent)
            def on_agent_complete(source: Any, event: AgentExecutionCompletedEvent) -> None:
                self._handle_agent_end(event, status="ok")

            @bus.on(AgentExecutionErrorEvent)
            def on_agent_error(source: Any, event: AgentExecutionErrorEvent) -> None:
                self._handle_agent_end(event, status="error")

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
                self._check_delegation_complete(event, status="ok")

            @bus.on(ToolUsageErrorEvent)
            def on_tool_error(source: Any, event: ToolUsageErrorEvent) -> None:
                self._handle_event(event)
                self._check_delegation_complete(event, status="error")

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

    def _register_memory_events(self, bus: Any) -> None:
        """Register memory, knowledge, and reasoning event handlers.
        
        These events are available in newer CrewAI versions and provide
        visibility into agent memory operations and reasoning processes.
        """
        # Memory query events
        try:
            from crewai.events.types.memory_events import (
                MemoryQueryStartedEvent,
                MemoryQueryCompletedEvent,
                MemoryQueryFailedEvent,
                MemorySaveStartedEvent,
                MemorySaveCompletedEvent,
                MemorySaveFailedEvent,
            )

            @bus.on(MemoryQueryStartedEvent)
            def on_memory_query_start(source: Any, event: MemoryQueryStartedEvent) -> None:
                self._handle_event(event)

            @bus.on(MemoryQueryCompletedEvent)
            def on_memory_query_complete(source: Any, event: MemoryQueryCompletedEvent) -> None:
                self._handle_event(event)

            @bus.on(MemoryQueryFailedEvent)
            def on_memory_query_failed(source: Any, event: MemoryQueryFailedEvent) -> None:
                self._handle_event(event)

            @bus.on(MemorySaveStartedEvent)
            def on_memory_save_start(source: Any, event: MemorySaveStartedEvent) -> None:
                self._handle_event(event)

            @bus.on(MemorySaveCompletedEvent)
            def on_memory_save_complete(source: Any, event: MemorySaveCompletedEvent) -> None:
                self._handle_event(event)

            @bus.on(MemorySaveFailedEvent)
            def on_memory_save_failed(source: Any, event: MemorySaveFailedEvent) -> None:
                self._handle_event(event)

        except ImportError:
            pass

        # Knowledge retrieval events
        try:
            from crewai.events.types.knowledge_events import (
                KnowledgeQueryStartedEvent,
                KnowledgeQueryCompletedEvent,
                KnowledgeQueryFailedEvent,
                KnowledgeRetrievalStartedEvent,
                KnowledgeRetrievalCompletedEvent,
            )

            @bus.on(KnowledgeQueryStartedEvent)
            def on_knowledge_query_start(source: Any, event: KnowledgeQueryStartedEvent) -> None:
                self._handle_event(event)

            @bus.on(KnowledgeQueryCompletedEvent)
            def on_knowledge_query_complete(source: Any, event: KnowledgeQueryCompletedEvent) -> None:
                self._handle_event(event)

            @bus.on(KnowledgeQueryFailedEvent)
            def on_knowledge_query_failed(source: Any, event: KnowledgeQueryFailedEvent) -> None:
                self._handle_event(event)

            @bus.on(KnowledgeRetrievalStartedEvent)
            def on_knowledge_retrieval_start(source: Any, event: KnowledgeRetrievalStartedEvent) -> None:
                self._handle_event(event)

            @bus.on(KnowledgeRetrievalCompletedEvent)
            def on_knowledge_retrieval_complete(source: Any, event: KnowledgeRetrievalCompletedEvent) -> None:
                self._handle_event(event)

        except ImportError:
            pass

        # Agent reasoning events
        try:
            from crewai.events.types.reasoning_events import (
                AgentReasoningStartedEvent,
                AgentReasoningCompletedEvent,
                AgentReasoningFailedEvent,
            )

            @bus.on(AgentReasoningStartedEvent)
            def on_reasoning_start(source: Any, event: AgentReasoningStartedEvent) -> None:
                self._handle_event(event)

            @bus.on(AgentReasoningCompletedEvent)
            def on_reasoning_complete(source: Any, event: AgentReasoningCompletedEvent) -> None:
                self._handle_event(event)

            @bus.on(AgentReasoningFailedEvent)
            def on_reasoning_failed(source: Any, event: AgentReasoningFailedEvent) -> None:
                self._handle_event(event)

        except ImportError:
            pass

    # =========================================================================
    # Flow Lifecycle Handlers (for multi-crew orchestration)
    # =========================================================================

    def _handle_flow_start(self, event: Any) -> None:
        """Handle flow start - establish flow-level span.
        
        When a Flow starts, use the existing run from init() and 
        set up the flow span as the parent for all method executions.
        """
        run = current_run()
        if not run:
            return

        # Cache run_id for thread fallback
        self._cache_run_id(run.run_id)

        # Create flow-level span as parent for all method executions
        flow_span_id = new_span_id()
        run.set_flow_span(flow_span_id)

        # Extract flow info
        flow_name = getattr(event, "flow_name", None) or "unknown_flow"
        inputs = getattr(event, "inputs", None)

        trace_event = evt_flow_start(
            run,
            flow_name=flow_name,
            span_id=flow_span_id,
            parent_span_id=run._root_span_id,
            inputs=inputs,
        )
        run.emit(trace_event)

    def _handle_flow_end(self, event: Any, status: str) -> None:
        """Handle flow completion."""
        run = self._get_run_with_fallback(event.__class__.__name__)
        if not run:
            return

        flow_name = getattr(event, "flow_name", None) or "unknown_flow"
        result = getattr(event, "result", None)
        state = getattr(event, "state", None)
        
        # Convert state to dict if it's a Pydantic model
        state_dict = None
        if state is not None:
            if hasattr(state, "model_dump"):
                state_dict = state.model_dump()
            elif hasattr(state, "dict"):
                state_dict = state.dict()
            elif isinstance(state, dict):
                state_dict = state

        flow_span_id = run.get_flow_span()
        
        trace_event = evt_flow_end(
            run,
            flow_name=flow_name,
            span_id=new_span_id(),
            parent_span_id=flow_span_id,
            status=status,
            result=result,
            state=state_dict,
        )
        run.emit(trace_event)

        # Clean up flow context
        run.clear_flow_span()
        run.clear_method_span()
        
        # Clear cached run_id when flow ends (only if not in a crew)
        if not run.get_crew_span():
            self._clear_cached_run_id()

    def _handle_method_start(self, event: Any) -> None:
        """Handle flow method execution start.
        
        Each method execution gets its own span, parented to the flow span.
        Crews executed within methods will be parented to the method span.
        """
        run = self._get_run_with_fallback(event.__class__.__name__)
        if not run:
            return

        flow_name = getattr(event, "flow_name", None) or "unknown_flow"
        method_name = getattr(event, "method_name", None) or "unknown_method"
        params = getattr(event, "params", None)
        state = getattr(event, "state", None)
        
        # Convert state to dict if it's a Pydantic model
        state_dict = None
        if state is not None:
            if hasattr(state, "model_dump"):
                state_dict = state.model_dump()
            elif hasattr(state, "dict"):
                state_dict = state.dict()
            elif isinstance(state, dict):
                state_dict = state

        # Create method span parented to flow span
        method_span_id = new_span_id()
        run.set_method_span(method_span_id)
        
        flow_span_id = run.get_flow_span()

        trace_event = evt_method_start(
            run,
            flow_name=flow_name,
            method_name=method_name,
            span_id=method_span_id,
            parent_span_id=flow_span_id,
            params=params,
            state=state_dict,
        )
        run.emit(trace_event)

    def _handle_method_end(self, event: Any, status: str) -> None:
        """Handle flow method execution completion."""
        run = self._get_run_with_fallback(event.__class__.__name__)
        if not run:
            return

        flow_name = getattr(event, "flow_name", None) or "unknown_flow"
        method_name = getattr(event, "method_name", None) or "unknown_method"
        result = getattr(event, "result", None)
        state = getattr(event, "state", None)
        error = getattr(event, "error", None)
        
        # Convert state to dict if it's a Pydantic model
        state_dict = None
        if state is not None:
            if hasattr(state, "model_dump"):
                state_dict = state.model_dump()
            elif hasattr(state, "dict"):
                state_dict = state.dict()
            elif isinstance(state, dict):
                state_dict = state
        
        error_str = str(error) if error else None

        method_span_id = run.get_method_span()

        trace_event = evt_method_end(
            run,
            flow_name=flow_name,
            method_name=method_name,
            span_id=new_span_id(),
            parent_span_id=method_span_id,
            status=status,
            result=result,
            state=state_dict,
            error=error_str,
        )
        run.emit(trace_event)

        # Clear method span (but keep flow span for next method)
        run.clear_method_span()
        # Clear crew span if one was created during this method
        run.clear_crew_span()

    # =========================================================================
    # Crew Lifecycle Handlers
    # =========================================================================

    def _handle_crew_start(self, event: Any) -> None:
        """Handle crew kickoff start - establish crew-level span.
        
        If inside a flow method, parents the crew to the method span.
        Otherwise, creates a fresh run to ensure clean sequence numbering.
        """
        # Check if we're inside a flow context first
        existing_run = current_run()
        in_flow = existing_run and existing_run.has_flow_context()
        
        # For standalone crew execution, create a fresh run
        # This ensures each crew gets seq starting at 1, matching CrewAI's batch isolation
        if not in_flow:
            import arzule_ingest
            arzule_ingest.new_run()
        
        run = current_run()
        if not run:
            return
        
        # Cache run_id for thread fallback (critical for async task support)
        self._cache_run_id(run.run_id)
        
        # Check if we're inside a flow context
        if run.has_flow_context():
            # Inside a flow - parent crew to the current method span
            crew_span_id = new_span_id()
            run.set_crew_span(crew_span_id)
            
            # Use method span as parent if available, otherwise flow span
            parent_span = run.get_method_span() or run.get_flow_span()
            
            trace_event = evt_from_crewai_event(run, event, parent_span_id=parent_span)
            trace_event["span_id"] = crew_span_id
            trace_event["attrs_compact"]["in_flow"] = True
            run.emit(trace_event)
        else:
            # Standalone crew execution - use existing run from init()
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

        # Determine parent span based on context
        if run.has_flow_context():
            parent_span = run.get_method_span() or run.get_flow_span()
        else:
            parent_span = run._root_span_id

        trace_event = evt_from_crewai_event(
            run, event,
            parent_span_id=parent_span,
        )
        trace_event["status"] = status
        if run.has_flow_context():
            trace_event["attrs_compact"]["in_flow"] = True
        run.emit(trace_event)

        # Clean up implicit handoff tracking state
        cleanup_run_tracking(run.run_id)

        # Only clear cached run_id if NOT in a flow (flow manages its own cleanup)
        if not run.has_flow_context():
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
        
        Also detects implicit context handoffs when this task depends on
        other tasks' outputs.
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

            # Detect implicit handoffs from context tasks
            # This captures agent-to-agent info flow through task dependencies
            implicit_handoff_keys = []
            sequential_handoff_key = None
            if task:
                # Check for explicit context dependencies
                implicit_handoff_keys = detect_task_context_handoff(
                    run, task, span_id=task_span_id
                )
                # Check for sequential agent transitions (different agent starting)
                sequential_handoff_key = detect_sequential_handoff(
                    run, task, span_id=task_span_id
                )

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
            
            # Track implicit handoffs for this task
            if implicit_handoff_keys:
                trace_event["attrs_compact"]["implicit_handoff_keys"] = implicit_handoff_keys
                trace_event["attrs_compact"]["context_task_count"] = len(implicit_handoff_keys)
            if sequential_handoff_key:
                trace_event["attrs_compact"]["sequential_handoff_key"] = sequential_handoff_key
            
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
        
        Also emits handoff.complete for any implicit context handoffs
        that targeted this task.
        """
        run = self._get_run_with_fallback(event.__class__.__name__)
        if not run:
            return

        task_key = _get_task_key(event)
        task = getattr(event, "task", None)

        if task_key:
            # Get task's root span before ending it
            task_span_id = run.get_task_root_span(task_key)
            parent_span_id = run.get_crew_span()

            # Check if this was an async task (has async context)
            async_id = run.get_async_id(task_key)
            
            # Emit handoff.complete for implicit context handoffs
            implicit_complete_count = 0
            if task:
                implicit_complete_count = emit_implicit_handoff_complete(
                    run, task, status=status, span_id=task_span_id
                )

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
            
            # Track implicit handoffs completed
            if implicit_complete_count > 0:
                trace_event["attrs_compact"]["implicit_handoffs_completed"] = implicit_complete_count
            
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
    # Agent Lifecycle Handlers (with thread-local agent tracking)
    # =========================================================================

    def _handle_agent_start(self, event: Any) -> None:
        """Handle agent execution start - set thread-local agent context.
        
        This is critical for proper attribution of LLM and tool events to agents.
        When an agent starts executing in a thread, we store its info so that
        subsequent events in the same thread can be attributed to it.
        """
        run = self._get_run_with_fallback(event.__class__.__name__)
        if not run:
            return

        # Extract and store agent info for this thread
        agent_info = extract_agent_info_from_event(event)
        if agent_info:
            run.set_current_agent(agent_info)

        # Emit the event as normal
        task_key = _get_task_key(event)
        agent_key = _get_agent_key(event)
        parent_span_id = run.get_task_parent_span(task_key=task_key, agent_key=agent_key)

        trace_event = evt_from_crewai_event(
            run, event,
            parent_span_id=parent_span_id,
            task_key=task_key,
        )

        if task_key:
            trace_event["attrs_compact"]["task_key"] = task_key
        if run.has_concurrent_tasks():
            trace_event["attrs_compact"]["concurrent_mode"] = True

        run.emit(trace_event)

    def _handle_agent_end(self, event: Any, status: str) -> None:
        """Handle agent execution end - clear thread-local agent context.
        
        Clears the agent context so that events after the agent finishes
        won't be incorrectly attributed to it.
        
        IMPORTANT: Agent context must be cleared even if run lookup fails,
        to prevent stale context from persisting in the thread.
        """
        run = self._get_run_with_fallback(event.__class__.__name__)
        if not run:
            # Still clear agent context even without a run to prevent stale attribution
            clear_current_agent_context()
            return

        # Emit the event first (while we still have context)
        task_key = _get_task_key(event)
        agent_key = _get_agent_key(event)
        parent_span_id = run.get_task_parent_span(task_key=task_key, agent_key=agent_key)

        trace_event = evt_from_crewai_event(
            run, event,
            parent_span_id=parent_span_id,
            task_key=task_key,
        )
        trace_event["status"] = status

        if task_key:
            trace_event["attrs_compact"]["task_key"] = task_key
        if run.has_concurrent_tasks():
            trace_event["attrs_compact"]["concurrent_mode"] = True

        run.emit(trace_event)

        # Clear agent context for this thread after emitting the event
        clear_current_agent_context()

    # =========================================================================
    # Generic Event Handler (for tool, LLM events)
    # =========================================================================

    def _handle_event(self, event: Any) -> None:
        """
        Convert and emit a CrewAI event as a trace event.

        For concurrent execution, this determines the correct parent span
        by looking up the task associated with the event's agent.
        
        For LLM and tool events that don't have agent context from CrewAI,
        injects the current thread-local agent so events are properly attributed.
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

        # Inject current agent if event doesn't have one
        # This handles LLM/tool events where CrewAI doesn't propagate agent context
        if trace_event.get("agent") is None:
            current_agent = run.get_current_agent()
            if current_agent:
                trace_event["agent"] = current_agent
                trace_event["attrs_compact"]["agent_injected"] = True

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

        # Extract tool_args - CrewAI uses 'tool_args' field
        # Also check fallback names for compatibility
        tool_input = getattr(event, "tool_args", None)
        if tool_input is None:
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
        # CrewAI provides agent_role directly on event, fallback to agent.role
        from_role = getattr(event, "agent_role", None)
        if not from_role:
            agent = getattr(event, "agent", None)
            from_role = getattr(agent, "role", None) if agent else None
        to_coworker = tool_input.get("coworker") or tool_input.get("to") or tool_input.get("agent")
        run._handoff_pending[handoff_key] = {
            "type": "delegation",  # Mark as delegation for tool-finish correlation
            "from_role": from_role,
            "to_coworker": to_coworker,
            "proposed_at": run.now(),
            "tool_name": tool_name,
            "tool_input": tool_input,
        }

        # Emit handoff.proposed with a new span_id
        maybe_emit_handoff_proposed(run, context, span_id=new_span_id())

    def _check_delegation_complete(self, event: Any, status: str) -> None:
        """
        Check if this tool completion is a delegation finishing.
        
        CrewAI's delegation tools work synchronously - the target agent executes
        inline and returns immediately. So we emit both handoff.ack and 
        handoff.complete when the delegation tool finishes.
        
        This handles the case where delegation doesn't create a new task,
        so we can't detect completion via task events.
        """
        run = self._get_run_with_fallback(event.__class__.__name__)
        if not run:
            return

        tool_name = getattr(event, "tool_name", None)
        if not is_delegation_tool(tool_name):
            return

        # Find pending delegation handoffs from the current agent
        # CrewAI provides agent_role directly on event, fallback to agent.role
        from_role = getattr(event, "agent_role", None)
        if not from_role:
            agent = getattr(event, "agent", None)
            from_role = getattr(agent, "role", None) if agent else None
        
        # Get result from the tool output (for semantic analysis)
        result = getattr(event, "result", None) or getattr(event, "output", None)
        result_summary = None
        if result:
            result_str = str(result)
            result_summary = result_str[:200] + "..." if len(result_str) > 200 else result_str

        # Find and complete pending delegation handoffs from this agent
        keys_to_remove = []
        for handoff_key, pending in list(run._handoff_pending.items()):
            if pending.get("type") != "delegation":
                continue
            if pending.get("from_role") != from_role:
                continue
            # Match by tool_name if available (helps with concurrent delegations)
            if pending.get("tool_name") and pending.get("tool_name") != tool_name:
                continue
            
            keys_to_remove.append(handoff_key)
            to_coworker = pending.get("to_coworker")
            
            # Emit handoff.ack (target agent acknowledged by starting execution)
            emit_handoff_ack(
                run=run,
                handoff_key=handoff_key,
                task_id=None,  # No task created for inline delegation
                agent_role=to_coworker,
                span_id=new_span_id(),
            )
            
            # Emit handoff.complete (delegation finished)
            # Include full result for semantic drift detection
            emit_handoff_complete(
                run=run,
                handoff_key=handoff_key,
                task_id=None,
                agent_role=to_coworker,
                status=status,
                result_summary=result_summary,
                result_payload=result,  # Full result for semantic analysis
                span_id=new_span_id(),
            )

        # Clean up completed handoffs
        for key in keys_to_remove:
            run._handoff_pending.pop(key, None)

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


def clear_listener_cache() -> None:
    """Clear the listener's cached run_id.
    
    Called by new_run() to prevent stale run_id from being used
    when callbacks arrive from background threads.
    """
    if _listener_instance is not None:
        _listener_instance._clear_cached_run_id()
