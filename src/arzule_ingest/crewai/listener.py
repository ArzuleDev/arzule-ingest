"""CrewAI event listener for lifecycle events (CrewAI 1.7.x API)."""

from __future__ import annotations

import threading
from typing import Any, Optional

from ..run import current_run
from ..logger import get_logger, log_event_dropped
from .handoff import (
    emit_handoff_ack,
    emit_handoff_complete,
    extract_handoff_key_from_text,
    is_delegation_tool,
    maybe_inject_handoff_key,
    maybe_emit_handoff_proposed,
)
from .implicit_handoff import (
    cleanup_run_tracking,
    detect_sequential_handoff,
    detect_task_context_handoff,
    emit_implicit_handoff_complete,
)
from .normalize import evt_from_crewai_event

# Global singleton to prevent GC
_listener_instance: Optional["ArzuleCrewAIListener"] = None


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
    
    Thread Safety:
    - Caches run_id for fallback when ContextVar fails in spawned threads
    - Uses _cached_run_id_lock for thread-safe access to the cached value
    """

    def __init__(self) -> None:
        """Initialize and register the listener."""
        self._setup_complete = False
        # Cache run_id for thread fallback when ContextVar fails
        self._cached_run_id: Optional[str] = None
        self._cached_run_id_lock = threading.Lock()
        # Track pending delegation tool calls for handoff correlation
        # Key: (run_id, tool_name, agent_role) -> {handoff_key, tool_input}
        self._pending_delegation_tools: dict[tuple[str, str, Optional[str]], dict[str, Any]] = {}
        self._pending_delegation_lock = threading.Lock()

    def _cache_run_id(self, run_id: str) -> None:
        """Cache the run_id for thread-safe fallback lookup.
        
        Called when we first see a valid run (crew/flow start). This allows
        spawned threads to recover the run via global registry.
        """
        with self._cached_run_id_lock:
            self._cached_run_id = run_id

    def _clear_cached_run_id(self) -> None:
        """Clear the cached run_id when starting a new run.
        
        Called by clear_listener_cache() from new_run() to prevent
        stale run_id from being used by background threads.
        """
        with self._cached_run_id_lock:
            self._cached_run_id = None

    def _get_cached_run_id(self) -> Optional[str]:
        """Get the cached run_id (thread-safe)."""
        with self._cached_run_id_lock:
            return self._cached_run_id

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

        logger = get_logger()
        logger.info("Setting up CrewAI event listeners")

        # Register all event handlers
        self._register_crew_events(crewai_event_bus)
        self._register_agent_events(crewai_event_bus)
        self._register_task_events(crewai_event_bus)
        self._register_tool_events(crewai_event_bus)
        self._register_llm_events(crewai_event_bus)
        self._register_flow_events(crewai_event_bus)
        self._register_memory_events(crewai_event_bus)
        self._register_knowledge_events(crewai_event_bus)
        self._register_a2a_events(crewai_event_bus)

        self._setup_complete = True
        logger.info("CrewAI event listeners setup complete")

    def _register_crew_events(self, bus: Any) -> None:
        """Register crew lifecycle event handlers."""
        logger = get_logger()
        try:
            from crewai.events.types.crew_events import (
                CrewKickoffCompletedEvent,
                CrewKickoffFailedEvent,
                CrewKickoffStartedEvent,
            )

            @bus.on(CrewKickoffStartedEvent)
            def on_crew_start(source: Any, event: CrewKickoffStartedEvent) -> None:
                # Ensure a run exists before handling crew start
                import arzule_ingest
                run_id = arzule_ingest.ensure_run()
                
                # Cache run_id for thread fallback (for concurrent tasks)
                if run_id:
                    self._cache_run_id(run_id)
                else:
                    logger.warning("ensure_run() returned None - SDK not initialized?")
                
                self._handle_event(event)

            @bus.on(CrewKickoffCompletedEvent)
            def on_crew_complete(source: Any, event: CrewKickoffCompletedEvent) -> None:
                self._handle_event(event)
                self._cleanup_implicit_handoff_tracking()

            @bus.on(CrewKickoffFailedEvent)
            def on_crew_failed(source: Any, event: CrewKickoffFailedEvent) -> None:
                self._handle_event(event)
                self._cleanup_implicit_handoff_tracking()

        except ImportError as e:
            logger.warning(f"Failed to import crew events: {e}")

    def _register_agent_events(self, bus: Any) -> None:
        """Register agent lifecycle event handlers."""
        logger = get_logger()
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


        except ImportError as e:
            logger.warning(f"Failed to import agent events: {e}")

    def _register_task_events(self, bus: Any) -> None:
        """Register task lifecycle event handlers."""
        logger = get_logger()
        try:
            from crewai.events.types.task_events import (
                TaskCompletedEvent,
                TaskFailedEvent,
                TaskStartedEvent,
            )
            @bus.on(TaskStartedEvent)
            def on_task_start(source: Any, event: TaskStartedEvent) -> None:
                self._handle_event(event)
                self._check_handoff_ack(event)
                self._detect_implicit_handoffs(event)

            @bus.on(TaskCompletedEvent)
            def on_task_complete(source: Any, event: TaskCompletedEvent) -> None:
                self._handle_event(event)
                self._check_handoff_complete(event, status="ok")
                self._complete_implicit_handoffs(event, status="ok")

            @bus.on(TaskFailedEvent)
            def on_task_failed(source: Any, event: TaskFailedEvent) -> None:
                self._handle_event(event)
                self._check_handoff_complete(event, status="error")
                self._complete_implicit_handoffs(event, status="error")


        except ImportError as e:
            logger.warning(f"Failed to import task events: {e}")

    def _register_tool_events(self, bus: Any) -> None:
        """Register tool usage event handlers."""
        logger = get_logger()
        try:
            from crewai.events.types.tool_usage_events import (
                ToolUsageFinishedEvent,
                ToolUsageStartedEvent,
                ToolUsageErrorEvent,
            )

            @bus.on(ToolUsageStartedEvent)
            def on_tool_start(source: Any, event: ToolUsageStartedEvent) -> None:
                self._handle_event(event)
                self._maybe_inject_delegation_handoff(event)

            @bus.on(ToolUsageFinishedEvent)
            def on_tool_end(source: Any, event: ToolUsageFinishedEvent) -> None:
                self._handle_event(event)
                self._maybe_emit_delegation_handoff(event)

            @bus.on(ToolUsageErrorEvent)
            def on_tool_error(source: Any, event: ToolUsageErrorEvent) -> None:
                self._handle_event(event)

        except ImportError as e:
            logger.warning(f"Failed to import tool events: {e}")

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

    def _register_flow_events(self, bus: Any) -> None:
        """Register flow lifecycle event handlers for multi-crew orchestration.
        
        Flow events track the execution of CrewAI Flows, which orchestrate
        multiple crews through a series of methods decorated with @start, @listen, etc.
        
        Key events:
        - FlowStartedEvent: When a flow begins execution
        - FlowFinishedEvent: When a flow completes (with final state)
        - MethodExecutionStartedEvent: When a flow method begins
        - MethodExecutionFinishedEvent: When a flow method completes
        - MethodExecutionFailedEvent: When a flow method fails
        """
        try:
            from crewai.events.types.flow_events import (
                FlowStartedEvent,
                FlowFinishedEvent,
                FlowCreatedEvent,
                MethodExecutionStartedEvent,
                MethodExecutionFinishedEvent,
                MethodExecutionFailedEvent,
            )

            @bus.on(FlowCreatedEvent)
            def on_flow_created(source: Any, event: FlowCreatedEvent) -> None:
                self._handle_event(event)

            @bus.on(FlowStartedEvent)
            def on_flow_start(source: Any, event: FlowStartedEvent) -> None:
                # Ensure a run exists before handling flow start
                import arzule_ingest
                run_id = arzule_ingest.ensure_run()
                
                # Cache run_id for thread fallback (for concurrent tasks)
                if run_id:
                    self._cache_run_id(run_id)
                else:
                    get_logger().warning("ensure_run() returned None - SDK not initialized?")
                
                self._handle_event(event)

            @bus.on(FlowFinishedEvent)
            def on_flow_end(source: Any, event: FlowFinishedEvent) -> None:
                self._handle_event(event)
                self._cleanup_implicit_handoff_tracking()

            @bus.on(MethodExecutionStartedEvent)
            def on_method_start(source: Any, event: MethodExecutionStartedEvent) -> None:
                self._handle_event(event)

            @bus.on(MethodExecutionFinishedEvent)
            def on_method_end(source: Any, event: MethodExecutionFinishedEvent) -> None:
                self._handle_event(event)

            @bus.on(MethodExecutionFailedEvent)
            def on_method_failed(source: Any, event: MethodExecutionFailedEvent) -> None:
                self._handle_event(event)

        except ImportError:
            # Flow events not available in this CrewAI version
            pass

        # Optional: Paused events (for human-in-the-loop flows)
        try:
            from crewai.events.types.flow_events import (
                FlowPausedEvent,
                MethodExecutionPausedEvent,
                HumanFeedbackRequestedEvent,
                HumanFeedbackReceivedEvent,
            )

            @bus.on(FlowPausedEvent)
            def on_flow_paused(source: Any, event: FlowPausedEvent) -> None:
                self._handle_event(event)

            @bus.on(MethodExecutionPausedEvent)
            def on_method_paused(source: Any, event: MethodExecutionPausedEvent) -> None:
                self._handle_event(event)

            @bus.on(HumanFeedbackRequestedEvent)
            def on_human_feedback_requested(source: Any, event: HumanFeedbackRequestedEvent) -> None:
                self._handle_event(event)

            @bus.on(HumanFeedbackReceivedEvent)
            def on_human_feedback_received(source: Any, event: HumanFeedbackReceivedEvent) -> None:
                self._handle_event(event)

        except ImportError:
            # HITL flow events not available in this CrewAI version
            pass

    def _register_memory_events(self, bus: Any) -> None:
        """Register memory event handlers.
        
        Memory events track agent memory operations (short-term, long-term, entity).
        Available in newer CrewAI versions.
        """
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
            # Memory events not available in this CrewAI version
            pass

    def _register_knowledge_events(self, bus: Any) -> None:
        """Register knowledge event handlers.
        
        Knowledge events track RAG/knowledge retrieval operations.
        Available in newer CrewAI versions with knowledge sources configured.
        """
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
            # Knowledge events not available in this CrewAI version
            pass

    def _register_a2a_events(self, bus: Any) -> None:
        """Register A2A (Agent-to-Agent) delegation event handlers.
        
        A2A events track delegation between agents, including:
        - Delegation start/complete
        - Multi-turn conversations
        - Messages sent/received between agents
        
        These events are critical for tracking delegation scenarios.
        """
        logger = get_logger()
        try:
            from crewai.events.types.a2a_events import (
                A2ADelegationStartedEvent,
                A2ADelegationCompletedEvent,
                A2AConversationStartedEvent,
                A2AConversationCompletedEvent,
                A2AMessageSentEvent,
                A2AResponseReceivedEvent,
            )

            @bus.on(A2ADelegationStartedEvent)
            def on_a2a_delegation_start(source: Any, event: A2ADelegationStartedEvent) -> None:
                self._handle_event(event)

            @bus.on(A2ADelegationCompletedEvent)
            def on_a2a_delegation_complete(source: Any, event: A2ADelegationCompletedEvent) -> None:
                self._handle_event(event)

            @bus.on(A2AConversationStartedEvent)
            def on_a2a_conversation_start(source: Any, event: A2AConversationStartedEvent) -> None:
                self._handle_event(event)

            @bus.on(A2AConversationCompletedEvent)
            def on_a2a_conversation_complete(source: Any, event: A2AConversationCompletedEvent) -> None:
                self._handle_event(event)

            @bus.on(A2AMessageSentEvent)
            def on_a2a_message_sent(source: Any, event: A2AMessageSentEvent) -> None:
                self._handle_event(event)

            @bus.on(A2AResponseReceivedEvent)
            def on_a2a_response_received(source: Any, event: A2AResponseReceivedEvent) -> None:
                self._handle_event(event)

        except ImportError:
            # A2A events not available in this CrewAI version
            pass

    def _handle_event(self, event: Any) -> None:
        """Convert and emit a CrewAI event as a trace event.
        
        Uses cached run_id for thread fallback when ContextVar fails
        (e.g., in CrewAI-spawned worker threads for concurrent tasks).
        """
        event_class = event.__class__.__name__
        cached_run_id = self._get_cached_run_id()
        run = current_run(run_id_hint=cached_run_id)
        if not run:
            log_event_dropped(
                reason="no_active_run_and_fallback_failed",
                event_class=event_class,
                extra={"cached_run_id": cached_run_id}
            )
            return

        trace_event = evt_from_crewai_event(run, event)
        run.emit(trace_event)

    def _maybe_inject_delegation_handoff(self, event: Any) -> None:
        """Inject handoff key for delegation tool calls and track for later correlation."""
        run = current_run(run_id_hint=self._get_cached_run_id())
        if not run:
            return

        tool_name = getattr(event, "tool_name", None)
        if not is_delegation_tool(tool_name):
            return

        # Create a context-like object for the handoff functions
        # The tool_input might be in tool_args or tool_input depending on CrewAI version
        tool_input = getattr(event, "tool_args", None)
        if tool_input is None:
            tool_input = getattr(event, "tool_input", None)
        if not isinstance(tool_input, dict):
            return

        agent = getattr(event, "agent", None)
        agent_role = getattr(agent, "role", None) if agent else None

        # Create context object matching what maybe_inject_handoff_key expects
        class ToolContext:
            pass

        context = ToolContext()
        context.tool_name = tool_name
        context.tool_input = tool_input
        context.agent = agent

        handoff_key = maybe_inject_handoff_key(run, context)
        
        # Store the pending delegation info for correlation with the finish event
        # The start and finish events are different instances, so we need to track this
        if handoff_key:
            lookup_key = (run.run_id, tool_name, agent_role)
            with self._pending_delegation_lock:
                self._pending_delegation_tools[lookup_key] = {
                    "handoff_key": handoff_key,
                    "tool_input": tool_input.copy(),  # Copy since we modified it
                    "agent": agent,
                }

    def _maybe_emit_delegation_handoff(self, event: Any) -> None:
        """Emit handoff.proposed event for delegation tool calls."""
        run = current_run(run_id_hint=self._get_cached_run_id())
        if not run:
            return

        tool_name = getattr(event, "tool_name", None)
        if not is_delegation_tool(tool_name):
            return

        agent = getattr(event, "agent", None)
        agent_role = getattr(agent, "role", None) if agent else None

        # Look up the pending delegation info stored during tool start
        # We can't rely on tool_input from this event because it's a different instance
        lookup_key = (run.run_id, tool_name, agent_role)
        with self._pending_delegation_lock:
            pending_info = self._pending_delegation_tools.pop(lookup_key, None)

        if not pending_info:
            return

        handoff_key = pending_info.get("handoff_key")
        tool_input = pending_info.get("tool_input")
        stored_agent = pending_info.get("agent")

        if not handoff_key or not tool_input:
            return

        # Create context object for maybe_emit_handoff_proposed
        class ToolContext:
            pass

        context = ToolContext()
        context.tool_name = tool_name
        context.tool_input = tool_input
        context.agent = stored_agent or agent

        maybe_emit_handoff_proposed(run, context, span_id=None)

    def _check_handoff_ack(self, event: Any) -> None:
        """Check if this task start is acknowledging a handoff."""
        run = current_run(run_id_hint=self._get_cached_run_id())
        if not run:
            return

        # Extract task description to look for handoff marker
        task = getattr(event, "task", None)
        if not task:
            return

        description = getattr(task, "description", None)
        handoff_key = extract_handoff_key_from_text(description)

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

    def _check_handoff_complete(self, event: Any, status: str) -> None:
        """Check if this task completion completes a handoff."""
        run = current_run(run_id_hint=self._get_cached_run_id())
        if not run:
            return

        task = getattr(event, "task", None)
        if not task:
            return

        description = getattr(task, "description", None)
        handoff_key = extract_handoff_key_from_text(description)

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

    def _detect_implicit_handoffs(self, event: Any) -> None:
        """Detect implicit handoffs when a task starts.
        
        Checks for:
        1. Context dependencies - when task.context includes other tasks
        2. Sequential transitions - when different agents run tasks in sequence
        
        These are detected separately from explicit delegation tool handoffs.
        """
        run = current_run(run_id_hint=self._get_cached_run_id())
        if not run:
            return

        task = getattr(event, "task", None)
        if not task:
            return

        # Detect context-based handoffs (task.context dependencies)
        detect_task_context_handoff(run, task)

        # Detect sequential agent transitions
        detect_sequential_handoff(run, task)

    def _complete_implicit_handoffs(self, event: Any, status: str) -> None:
        """Complete any implicit handoffs when a task finishes.
        
        Emits handoff.complete events for context and sequential handoffs
        that targeted this task.
        """
        run = current_run(run_id_hint=self._get_cached_run_id())
        if not run:
            return

        task = getattr(event, "task", None)
        if not task:
            return

        emit_implicit_handoff_complete(run, task, status=status)

    def _cleanup_implicit_handoff_tracking(self) -> None:
        """Clean up implicit handoff tracking when a crew/flow completes.
        
        Clears the last completed task tracking to prevent stale data
        from affecting subsequent runs.
        """
        run = current_run(run_id_hint=self._get_cached_run_id())
        if run:
            cleanup_run_tracking(run.run_id)


def get_listener() -> ArzuleCrewAIListener:
    """Get or create the global listener singleton."""
    global _listener_instance
    if _listener_instance is None:
        _listener_instance = ArzuleCrewAIListener()
    return _listener_instance


def clear_listener_cache() -> None:
    """Clear the listener's cached run_id.
    
    Called by new_run() to prevent stale run_id from being used
    when callbacks arrive from background threads that don't have
    the ContextVar set.
    
    This is CRITICAL for preventing context corruption when creating
    multiple sequential runs with concurrent task execution.
    """
    if _listener_instance is not None:
        _listener_instance._clear_cached_run_id()
