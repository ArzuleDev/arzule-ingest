"""CrewAI event listener for lifecycle events (CrewAI 1.7.x API)."""

from __future__ import annotations

from typing import Any, Optional

from ..run import current_run
from .handoff import emit_handoff_ack, emit_handoff_complete, extract_handoff_key_from_text
from .normalize import evt_from_crewai_event

# Global singleton to prevent GC
_listener_instance: Optional["ArzuleCrewAIListener"] = None


class ArzuleCrewAIListener:
    """
    Event listener for CrewAI lifecycle events.

    Listens to crew, agent, and task lifecycle events and emits trace events.
    Also handles handoff acknowledgment and completion detection.

    NOTE: CrewAI event listeners can be garbage collected if not kept in memory.
    This class maintains a global singleton to prevent that.
    """

    def __init__(self) -> None:
        """Initialize and register the listener."""
        self._setup_complete = False

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
                self._handle_event(event)

            @bus.on(CrewKickoffCompletedEvent)
            def on_crew_complete(source: Any, event: CrewKickoffCompletedEvent) -> None:
                self._handle_event(event)

            @bus.on(CrewKickoffFailedEvent)
            def on_crew_failed(source: Any, event: CrewKickoffFailedEvent) -> None:
                self._handle_event(event)

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
                self._handle_event(event)
                self._check_handoff_ack(event)

            @bus.on(TaskCompletedEvent)
            def on_task_complete(source: Any, event: TaskCompletedEvent) -> None:
                self._handle_event(event)
                self._check_handoff_complete(event, status="ok")

            @bus.on(TaskFailedEvent)
            def on_task_failed(source: Any, event: TaskFailedEvent) -> None:
                self._handle_event(event)
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

    def _handle_event(self, event: Any) -> None:
        """Convert and emit a CrewAI event as a trace event."""
        run = current_run()
        if not run:
            return

        trace_event = evt_from_crewai_event(run, event)
        run.emit(trace_event)

    def _check_handoff_ack(self, event: Any) -> None:
        """Check if this task start is acknowledging a handoff."""
        run = current_run()
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
        run = current_run()
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


def get_listener() -> ArzuleCrewAIListener:
    """Get or create the global listener singleton."""
    global _listener_instance
    if _listener_instance is None:
        _listener_instance = ArzuleCrewAIListener()
    return _listener_instance
