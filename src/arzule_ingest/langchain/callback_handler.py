"""LangChain callback handler for Arzule observability.

Thread Safety:
- Caches run_id for fallback when ContextVar fails in spawned threads
- Uses global registry lookup when ContextVar returns None
"""

from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from uuid import UUID

from ..logger import log_event_dropped
from ..run import current_run
from ..ids import new_span_id
from .normalize import (
    evt_llm_start,
    evt_llm_end,
    evt_llm_error,
    evt_chain_start,
    evt_chain_end,
    evt_chain_error,
    evt_tool_start,
    evt_tool_end,
    evt_tool_error,
    evt_agent_action,
    evt_agent_finish,
    evt_retriever_start,
    evt_retriever_end,
    evt_retriever_error,
)
from .handoff import detect_handoff_from_agent_action

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.documents import Document
    from langchain_core.outputs import LLMResult
    from ..run import ArzuleRun

# Try to import BaseCallbackHandler, fall back to standalone class
try:
    from langchain_core.callbacks.base import BaseCallbackHandler

    _BASE_CLASS = BaseCallbackHandler
except ImportError:
    # LangChain not installed, use object as base
    _BASE_CLASS = object  # type: ignore


class ArzuleLangChainHandler(_BASE_CLASS):
    """
    Callback handler for LangChain that emits trace events to Arzule.

    This handler captures LLM calls, chain executions, tool invocations,
    agent actions, and retriever operations. It maintains span context
    for proper parent-child relationships.

    Usage:
        from arzule_ingest.langchain import instrument_langchain

        handler = instrument_langchain()
        
        # Pass to LangChain components
        llm = ChatOpenAI(callbacks=[handler])
        
        # Or use with invoke config
        chain.invoke(input, config={"callbacks": [handler]})
    """

    def __init__(
        self,
        enable_llm: bool = True,
        enable_chain: bool = True,
        enable_tool: bool = True,
        enable_agent: bool = True,
        enable_retriever: bool = True,
    ) -> None:
        """Initialize the handler with feature flags."""
        # Call parent init if inheriting from BaseCallbackHandler
        if _BASE_CLASS is not object:
            super().__init__()

        self.enable_llm = enable_llm
        self.enable_chain = enable_chain
        self.enable_tool = enable_tool
        self.enable_agent = enable_agent
        self.enable_retriever = enable_retriever

        # Track run_id -> span_id mapping for correlating start/end
        self._run_spans: Dict[str, str] = {}
        
        # Cached run_id for thread fallback when ContextVar fails
        self._cached_run_id: Optional[str] = None
        self._cached_run_id_lock = threading.Lock()

    # =========================================================================
    # Run Context Fallback (for spawned threads)
    # =========================================================================

    def _cache_run_id(self, run_id: str) -> None:
        """Cache the run_id for thread-safe fallback lookup."""
        with self._cached_run_id_lock:
            self._cached_run_id = run_id

    def _clear_cached_run_id(self) -> None:
        """Clear the cached run_id."""
        with self._cached_run_id_lock:
            self._cached_run_id = None

    def _get_cached_run_id(self) -> Optional[str]:
        """Get the cached run_id (thread-safe)."""
        with self._cached_run_id_lock:
            return self._cached_run_id

    def _get_run_with_fallback(self, callback_name: str) -> Optional["ArzuleRun"]:
        """Get run from ContextVar, falling back to cached run_id or ensure_run().
        
        Args:
            callback_name: Name of the callback (for logging if dropped)
            
        Returns:
            The ArzuleRun instance, or None if not recoverable
        """
        # Try ContextVar first
        run = current_run()
        if run:
            # Update cache when we have a valid run
            self._cache_run_id(run.run_id)
            return run
        
        # Fallback: try global registry with cached run_id
        cached_id = self._get_cached_run_id()
        if cached_id:
            run = current_run(run_id_hint=cached_id)
            if run:
                return run
        
        # LangChain-specific: ensure a run exists (lazy creation)
        # This is needed when LangChain is used standalone (not via CrewAI)
        # or when LangGraph callbacks fire before CrewAI starts
        try:
            import arzule_ingest
            run_id = arzule_ingest.ensure_run()
            if run_id:
                run = current_run(run_id_hint=run_id)
                if run:
                    self._cache_run_id(run.run_id)
                    return run
        except Exception:
            pass
        
        # Log the drop (only if we had a cached_id, meaning we were expecting a run)
        if cached_id:
            log_event_dropped(
                reason="no_active_run_and_fallback_failed",
                event_class=f"langchain.{callback_name}",
                extra={"cached_run_id": cached_id}
            )
        return None

    def _get_run_key(self, run_id: UUID) -> str:
        """Convert UUID to string key."""
        return str(run_id)

    def _start_span(self, run_id: UUID) -> str:
        """Create and track a new span for a run."""
        span_id = new_span_id()
        self._run_spans[self._get_run_key(run_id)] = span_id
        return span_id

    def _end_span(self, run_id: UUID) -> Optional[str]:
        """Get and remove the span for a run."""
        return self._run_spans.pop(self._get_run_key(run_id), None)

    def _get_span(self, run_id: UUID) -> Optional[str]:
        """Get the span for a run without removing it."""
        return self._run_spans.get(self._get_run_key(run_id))

    # =========================================================================
    # LLM Callbacks
    # =========================================================================

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts running."""
        if not self.enable_llm:
            return

        run = self._get_run_with_fallback("on_llm_start")
        if not run:
            return

        span_id = self._start_span(run_id)
        parent_span = self._get_span(parent_run_id) if parent_run_id else None

        evt = evt_llm_start(
            run=run,
            span_id=span_id,
            parent_span_id=parent_span or run.current_parent_span_id(),
            serialized=serialized,
            prompts=prompts,
            tags=tags,
            metadata=metadata,
        )
        run.emit(evt)

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM generates a new token (streaming)."""
        # Skip token-level events to avoid noise - can be enabled if needed
        pass

    def on_llm_end(
        self,
        response: "LLMResult",
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM finishes running."""
        if not self.enable_llm:
            return

        run = self._get_run_with_fallback("on_llm_end")
        if not run:
            return

        span_id = self._end_span(run_id)
        parent_span = self._get_span(parent_run_id) if parent_run_id else None

        evt = evt_llm_end(
            run=run,
            span_id=span_id,
            parent_span_id=parent_span or run.current_parent_span_id(),
            response=response,
        )
        run.emit(evt)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors."""
        if not self.enable_llm:
            return

        run = self._get_run_with_fallback("on_llm_error")
        if not run:
            return

        span_id = self._end_span(run_id)
        parent_span = self._get_span(parent_run_id) if parent_run_id else None

        evt = evt_llm_error(
            run=run,
            span_id=span_id,
            parent_span_id=parent_span or run.current_parent_span_id(),
            error=error,
        )
        run.emit(evt)

    # =========================================================================
    # Chain Callbacks
    # =========================================================================

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain starts running."""
        if not self.enable_chain:
            return

        run = self._get_run_with_fallback("on_chain_start")
        if not run:
            return

        span_id = self._start_span(run_id)
        parent_span = self._get_span(parent_run_id) if parent_run_id else None

        evt = evt_chain_start(
            run=run,
            span_id=span_id,
            parent_span_id=parent_span or run.current_parent_span_id(),
            serialized=serialized,
            inputs=inputs,
            tags=tags,
            metadata=metadata,
        )
        run.emit(evt)

        # Push span onto stack for nested operations
        run.push_span(span_id)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain finishes running."""
        if not self.enable_chain:
            return

        run = self._get_run_with_fallback("on_chain_end")
        if not run:
            return

        span_id = self._end_span(run_id)
        parent_span = self._get_span(parent_run_id) if parent_run_id else None

        # Pop span from stack
        run.pop_span()

        evt = evt_chain_end(
            run=run,
            span_id=span_id,
            parent_span_id=parent_span or run.current_parent_span_id(),
            outputs=outputs,
        )
        run.emit(evt)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain errors."""
        if not self.enable_chain:
            return

        run = self._get_run_with_fallback("on_chain_error")
        if not run:
            return

        span_id = self._end_span(run_id)
        parent_span = self._get_span(parent_run_id) if parent_run_id else None

        # Pop span from stack
        run.pop_span()

        evt = evt_chain_error(
            run=run,
            span_id=span_id,
            parent_span_id=parent_span or run.current_parent_span_id(),
            error=error,
        )
        run.emit(evt)

    # =========================================================================
    # Tool Callbacks
    # =========================================================================

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool starts running."""
        if not self.enable_tool:
            return

        run = self._get_run_with_fallback("on_tool_start")
        if not run:
            return

        span_id = self._start_span(run_id)
        parent_span = self._get_span(parent_run_id) if parent_run_id else None

        evt = evt_tool_start(
            run=run,
            span_id=span_id,
            parent_span_id=parent_span or run.current_parent_span_id(),
            serialized=serialized,
            input_str=input_str,
            tags=tags,
            metadata=metadata,
        )
        run.emit(evt)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool finishes running."""
        if not self.enable_tool:
            return

        run = self._get_run_with_fallback("on_tool_end")
        if not run:
            return

        span_id = self._end_span(run_id)
        parent_span = self._get_span(parent_run_id) if parent_run_id else None

        evt = evt_tool_end(
            run=run,
            span_id=span_id,
            parent_span_id=parent_span or run.current_parent_span_id(),
            output=output,
        )
        run.emit(evt)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool errors."""
        if not self.enable_tool:
            return

        run = self._get_run_with_fallback("on_tool_error")
        if not run:
            return

        span_id = self._end_span(run_id)
        parent_span = self._get_span(parent_run_id) if parent_run_id else None

        evt = evt_tool_error(
            run=run,
            span_id=span_id,
            parent_span_id=parent_span or run.current_parent_span_id(),
            error=error,
        )
        run.emit(evt)

    # =========================================================================
    # Agent Callbacks
    # =========================================================================

    def on_agent_action(
        self,
        action: "AgentAction",
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent takes an action."""
        if not self.enable_agent:
            return

        run = self._get_run_with_fallback("on_agent_action")
        if not run:
            return

        span_id = self._start_span(run_id)
        parent_span = self._get_span(parent_run_id) if parent_run_id else None

        evt = evt_agent_action(
            run=run,
            span_id=span_id,
            parent_span_id=parent_span or run.current_parent_span_id(),
            action=action,
            tags=tags,
            metadata=metadata,
        )
        run.emit(evt)

        # Check for handoff patterns
        detect_handoff_from_agent_action(run, action, span_id)

    def on_agent_finish(
        self,
        finish: "AgentFinish",
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent finishes."""
        if not self.enable_agent:
            return

        run = self._get_run_with_fallback("on_agent_finish")
        if not run:
            return

        span_id = self._end_span(run_id)
        parent_span = self._get_span(parent_run_id) if parent_run_id else None

        evt = evt_agent_finish(
            run=run,
            span_id=span_id,
            parent_span_id=parent_span or run.current_parent_span_id(),
            finish=finish,
            tags=tags,
            metadata=metadata,
        )
        run.emit(evt)

    # =========================================================================
    # Retriever Callbacks
    # =========================================================================

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when retriever starts running."""
        if not self.enable_retriever:
            return

        run = self._get_run_with_fallback("on_retriever_start")
        if not run:
            return

        span_id = self._start_span(run_id)
        parent_span = self._get_span(parent_run_id) if parent_run_id else None

        evt = evt_retriever_start(
            run=run,
            span_id=span_id,
            parent_span_id=parent_span or run.current_parent_span_id(),
            serialized=serialized,
            query=query,
            tags=tags,
            metadata=metadata,
        )
        run.emit(evt)

    def on_retriever_end(
        self,
        documents: List["Document"],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when retriever finishes running."""
        if not self.enable_retriever:
            return

        run = self._get_run_with_fallback("on_retriever_end")
        if not run:
            return

        span_id = self._end_span(run_id)
        parent_span = self._get_span(parent_run_id) if parent_run_id else None

        evt = evt_retriever_end(
            run=run,
            span_id=span_id,
            parent_span_id=parent_span or run.current_parent_span_id(),
            documents=documents,
        )
        run.emit(evt)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when retriever errors."""
        if not self.enable_retriever:
            return

        run = self._get_run_with_fallback("on_retriever_error")
        if not run:
            return

        span_id = self._end_span(run_id)
        parent_span = self._get_span(parent_run_id) if parent_run_id else None

        evt = evt_retriever_error(
            run=run,
            span_id=span_id,
            parent_span_id=parent_span or run.current_parent_span_id(),
            error=error,
        )
        run.emit(evt)

