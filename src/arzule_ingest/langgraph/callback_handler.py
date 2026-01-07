"""LangGraph callback handler for Arzule observability.

Since LangGraph is built on LangChain and uses BaseCallbackHandler,
this handler extends LangChain's callback system to capture LangGraph-specific events.
"""

from __future__ import annotations

import sys
import threading
import uuid as uuid_module
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID

from ..logger import log_event_dropped
from ..run import current_run
from ..ids import new_span_id
from .normalize import (
    evt_graph_start,
    evt_graph_end,
    evt_node_start,
    evt_node_end,
    evt_checkpoint_save,
    evt_checkpoint_load,
    evt_task_start,
    evt_task_end,
    evt_llm_start,
    evt_llm_end,
    evt_tool_start,
    evt_tool_end,
    evt_handoff_proposed,
    evt_handoff_ack,
    evt_handoff_complete,
    evt_parallel_fanout,
    evt_parallel_fanin,
)

if TYPE_CHECKING:
    from langchain_core.outputs import LLMResult
    from ..run import ArzuleRun

# Try to import BaseCallbackHandler
try:
    from langchain_core.callbacks.base import BaseCallbackHandler

    _BASE_CLASS = BaseCallbackHandler
except ImportError:
    # LangChain/LangGraph not installed, use object as base
    _BASE_CLASS = object  # type: ignore


class ArzuleLangGraphHandler(_BASE_CLASS):
    """
    Callback handler for LangGraph that emits trace events to Arzule.

    This handler captures:
    - Graph execution start/end
    - Node execution start/end
    - Checkpoint save/load operations
    - Task execution (parallel nodes)
    - Nested LLM calls (from LangChain)
    - Nested tool calls

    Usage:
        from arzule_ingest.langgraph import instrument_langgraph
        from langgraph.graph import StateGraph

        handler = instrument_langgraph()
        
        graph = StateGraph(State)
        graph.add_node("node_a", node_a_func)
        compiled = graph.compile()
        
        # Use with invoke config
        result = compiled.invoke(input, config={"callbacks": [handler]})
    """

    def __init__(
        self,
        enable_graph: bool = True,
        enable_node: bool = True,
        enable_llm: bool = True,
        enable_tool: bool = True,
        enable_checkpoint: bool = True,
    ) -> None:
        """Initialize the handler with feature flags."""
        # Call parent init if inheriting from BaseCallbackHandler
        if _BASE_CLASS is not object:
            super().__init__()

        self.enable_graph = enable_graph
        self.enable_node = enable_node
        self.enable_llm = enable_llm
        self.enable_tool = enable_tool
        self.enable_checkpoint = enable_checkpoint

        # Track run_id -> span_id mapping for correlating start/end
        self._run_spans: Dict[str, str] = {}
        
        # Track graph execution context
        self._graph_names: Dict[str, str] = {}  # run_id -> graph_name
        self._node_names: Dict[str, str] = {}  # run_id -> node_name
        self._node_metadata: Dict[str, Dict[str, Any]] = {}  # run_id -> metadata (for parallel tracking)
        self._tool_names: Dict[str, str] = {}  # run_id -> tool_name
        
        # Parallel execution tracking
        # step_key (graph_run_id:step) -> list of node run_ids executing in that step
        self._parallel_steps: Dict[str, List[str]] = {}
        
        # Handoff tracking for multi-agent coordination
        # handoff_key -> {from_node, to_node, proposed_span_id}
        self._handoff_pending: Dict[str, Dict[str, Any]] = {}
        # node_name -> [handoff_keys] - maps target nodes to pending handoffs
        self._handoff_targets: Dict[str, List[str]] = {}
        
        # Cached run_id for thread fallback
        self._cached_run_id: Optional[str] = None
        self._cached_run_id_lock = threading.Lock()

    # =========================================================================
    # Run Context Management
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
        
        Unlike CrewAI which has explicit hooks for crew start, LangGraph callbacks
        may fire before a run is created. This method ensures a run exists by
        calling ensure_run() if needed.
        """
        # Try ContextVar first
        run = current_run()
        if run:
            self._cache_run_id(run.run_id)
            return run
        
        # Fallback: try global registry with cached run_id
        cached_id = self._get_cached_run_id()
        if cached_id:
            run = current_run(run_id_hint=cached_id)
            if run:
                return run
        
        # LangGraph-specific: ensure a run exists (lazy creation like CrewAI)
        # This is needed because LangGraph doesn't have explicit "graph started" hooks
        # that we can intercept before callbacks fire
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
        
        # Log the drop
        if cached_id:
            log_event_dropped(
                reason="no_active_run_and_fallback_failed",
                event_class=f"langgraph.{callback_name}",
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

    def _end_span(self, run_id: UUID) -> tuple[str, Optional[str]]:
        """Create a new span for end event, returning (new_span_id, parent_start_span_id).
        
        End events get their own span_id to prevent the database upsert from
        overwriting start events (which share run_id + span_id as the conflict key).
        The start event's span_id is returned as parent for correlation.
        """
        key = self._get_run_key(run_id)
        start_span_id = self._run_spans.pop(key, None)
        end_span_id = new_span_id()
        return end_span_id, start_span_id

    def _get_span(self, run_id: UUID) -> Optional[str]:
        """Get the span for a run without removing it."""
        return self._run_spans.get(self._get_run_key(run_id))

    # =========================================================================
    # Chain Callbacks (used by LangGraph for graph/node execution)
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
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain starts running.
        
        In LangGraph, this is triggered for:
        - Graph execution (name="LangGraph", no langgraph_node in metadata)
        - Node execution (name=<node_name>, metadata contains langgraph_node)
        
        Note: LangGraph passes serialized=None, so we detect graph/node by
        examining the name parameter and metadata dict.
        """
        run = self._get_run_with_fallback("on_chain_start")
        if not run:
            return

        # Determine if this is a LangGraph graph or node based on metadata
        # LangGraph passes serialized=None, but provides useful metadata
        chain_type = self._get_chain_type_from_metadata(name, metadata, tags, serialized)
        
        # Skip if this is not a LangGraph-specific chain
        if chain_type == "other":
            return
        
        # Extract the chain name from metadata or name parameter
        chain_name = self._extract_chain_name_from_metadata(name, metadata, serialized)
        
        span_id = self._start_span(run_id)
        parent_span = self._get_span(parent_run_id) if parent_run_id else None

        if chain_type == "graph" and self.enable_graph:
            # This is a graph execution
            self._graph_names[self._get_run_key(run_id)] = chain_name
            evt = evt_graph_start(
                run=run,
                span_id=span_id,
                parent_span_id=parent_span or run.current_parent_span_id(),
                graph_name=chain_name,
                input_data=inputs,
                metadata=metadata,
            )
            run.emit(evt)
            
        elif chain_type == "node" and self.enable_node:
            # This is a node execution
            run_key = self._get_run_key(run_id)
            self._node_names[run_key] = chain_name
            self._node_metadata[run_key] = metadata or {}  # Store for node_end
            
            # Track parallel execution by step
            # Nodes in the same step (superstep) execute in parallel
            if metadata and "langgraph_step" in metadata and parent_run_id:
                step_key = f"{self._get_run_key(parent_run_id)}:{metadata['langgraph_step']}"
                if step_key not in self._parallel_steps:
                    self._parallel_steps[step_key] = []
                self._parallel_steps[step_key].append(run_key)
            
            # Detect implicit handoffs from langgraph_triggers metadata
            # This catches handoffs from conditional edges where the source node
            # doesn't explicitly set next_agent in its output
            self._maybe_emit_implicit_handoff_proposed(run, chain_name, inputs, metadata, span_id, parent_span)
            
            # Check for handoff.ack if this node was a handoff target
            self._maybe_emit_handoff_ack(run, chain_name, span_id, parent_span)
            
            evt = evt_node_start(
                run=run,
                span_id=span_id,
                parent_span_id=parent_span or run.current_parent_span_id(),
                node_name=chain_name,
                input_data=inputs,
                metadata=metadata,
            )
            run.emit(evt)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain ends successfully."""
        run = self._get_run_with_fallback("on_chain_end")
        if not run:
            return

        span_id, start_span_id = self._end_span(run_id)
        # Use start_span_id as parent if available (links end to start event)
        parent_span = start_span_id or (self._get_span(parent_run_id) if parent_run_id else None)
        run_key = self._get_run_key(run_id)

        # Determine if this was a graph or node
        if run_key in self._graph_names and self.enable_graph:
            graph_name = self._graph_names.pop(run_key)
            evt = evt_graph_end(
                run=run,
                span_id=span_id,
                parent_span_id=parent_span or run.current_parent_span_id(),
                graph_name=graph_name,
                output_data=outputs,
            )
            run.emit(evt)
            
        elif run_key in self._node_names and self.enable_node:
            node_name = self._node_names.pop(run_key)
            node_metadata = self._node_metadata.pop(run_key, None)
            
            # Check for handoff.complete if this node was a handoff target
            self._maybe_emit_handoff_complete(run, node_name, outputs, span_id, parent_span)
            
            evt = evt_node_end(
                run=run,
                span_id=span_id,
                parent_span_id=parent_span or run.current_parent_span_id(),
                node_name=node_name,
                output_data=outputs,
                metadata=node_metadata,
            )
            run.emit(evt)
            
            # Check if output contains Send objects for parallel fan-out detection
            self._maybe_emit_parallel_fanout(run, node_name, outputs, span_id, parent_span)
            
            # Check if output contains a Command for handoff detection
            self._maybe_emit_handoff_proposed(run, node_name, outputs, span_id, parent_span)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain encounters an error."""
        run = self._get_run_with_fallback("on_chain_error")
        if not run:
            return

        span_id, start_span_id = self._end_span(run_id)
        parent_span = start_span_id or (self._get_span(parent_run_id) if parent_run_id else None)
        run_key = self._get_run_key(run_id)

        # Determine if this was a graph or node
        if run_key in self._graph_names and self.enable_graph:
            graph_name = self._graph_names.pop(run_key)
            evt = evt_graph_end(
                run=run,
                span_id=span_id,
                parent_span_id=parent_span or run.current_parent_span_id(),
                graph_name=graph_name,
                output_data=None,
                error=error,
            )
            run.emit(evt)
            
        elif run_key in self._node_names and self.enable_node:
            node_name = self._node_names.pop(run_key)
            node_metadata = self._node_metadata.pop(run_key, None)
            
            # Check for handoff.complete with error if this node was a handoff target
            self._maybe_emit_handoff_complete(run, node_name, None, span_id, parent_span, error=error)
            
            evt = evt_node_end(
                run=run,
                span_id=span_id,
                parent_span_id=parent_span or run.current_parent_span_id(),
                node_name=node_name,
                output_data=None,
                error=error,
                metadata=node_metadata,
            )
            run.emit(evt)

    # =========================================================================
    # LLM Callbacks (from nested LangChain calls)
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
        model_name = self._extract_llm_name(serialized)

        evt = evt_llm_start(
            run=run,
            span_id=span_id,
            parent_span_id=parent_span or run.current_parent_span_id(),
            model_name=model_name,
            prompts=prompts,
            metadata=metadata,
        )
        run.emit(evt)

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

        span_id, start_span_id = self._end_span(run_id)
        parent_span = start_span_id or (self._get_span(parent_run_id) if parent_run_id else None)

        # Try to extract model name from response metadata
        model_name = "unknown"
        if hasattr(response, "llm_output") and response.llm_output:
            model_name = response.llm_output.get("model_name", "unknown")

        evt = evt_llm_end(
            run=run,
            span_id=span_id,
            parent_span_id=parent_span or run.current_parent_span_id(),
            model_name=model_name,
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
        """Called when LLM encounters an error."""
        if not self.enable_llm:
            return

        run = self._get_run_with_fallback("on_llm_error")
        if not run:
            return

        span_id, start_span_id = self._end_span(run_id)
        parent_span = start_span_id or (self._get_span(parent_run_id) if parent_run_id else None)

        evt = evt_llm_end(
            run=run,
            span_id=span_id,
            parent_span_id=parent_span or run.current_parent_span_id(),
            model_name="unknown",
            response=None,
            error=error,
        )
        run.emit(evt)

    # =========================================================================
    # Tool Callbacks (from nested tool calls)
    # =========================================================================

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts running."""
        if not self.enable_tool:
            return

        run = self._get_run_with_fallback("on_tool_start")
        if not run:
            return

        span_id = self._start_span(run_id)
        parent_span = self._get_span(parent_run_id) if parent_run_id else None
        tool_name = serialized.get("name", "unknown")
        
        # Store tool name for retrieval in on_tool_end/on_tool_error
        self._tool_names[self._get_run_key(run_id)] = tool_name

        evt = evt_tool_start(
            run=run,
            span_id=span_id,
            parent_span_id=parent_span or run.current_parent_span_id(),
            tool_name=tool_name,
            tool_input=input_str,
        )
        run.emit(evt)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool finishes running."""
        if not self.enable_tool:
            return

        run = self._get_run_with_fallback("on_tool_end")
        if not run:
            return

        span_id, start_span_id = self._end_span(run_id)
        parent_span = start_span_id or (self._get_span(parent_run_id) if parent_run_id else None)
        
        # Retrieve stored tool name
        run_key = self._get_run_key(run_id)
        tool_name = self._tool_names.pop(run_key, "unknown")

        evt = evt_tool_end(
            run=run,
            span_id=span_id,
            parent_span_id=parent_span or run.current_parent_span_id(),
            tool_name=tool_name,
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
        """Called when a tool encounters an error."""
        if not self.enable_tool:
            return

        run = self._get_run_with_fallback("on_tool_error")
        if not run:
            return

        span_id, start_span_id = self._end_span(run_id)
        parent_span = start_span_id or (self._get_span(parent_run_id) if parent_run_id else None)
        
        # Retrieve stored tool name
        run_key = self._get_run_key(run_id)
        tool_name = self._tool_names.pop(run_key, "unknown")

        evt = evt_tool_end(
            run=run,
            span_id=span_id,
            parent_span_id=parent_span or run.current_parent_span_id(),
            tool_name=tool_name,
            output=None,
            error=error,
        )
        run.emit(evt)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_chain_type_from_metadata(
        self,
        name: Optional[str],
        metadata: Optional[Dict[str, Any]],
        tags: Optional[List[str]],
        serialized: Optional[Dict[str, Any]],
    ) -> str:
        """Determine if this is a graph, node, or other chain.
        
        LangGraph passes serialized=None but provides useful metadata:
        - Graph execution: name="LangGraph", no langgraph_node in metadata
        - Node execution: metadata contains langgraph_node, langgraph_step, etc.
        
        Returns:
            "graph" - LangGraph graph execution
            "node" - LangGraph node execution
            "other" - Regular LangChain chain (should be handled by LangChain handler)
        """
        metadata = metadata or {}
        tags = tags or []
        
        # Check if this is a LangGraph node by metadata
        # LangGraph sets langgraph_node in metadata for node executions
        if "langgraph_node" in metadata:
            return "node"
        
        # Check for other LangGraph metadata keys that indicate node execution
        # These are set by LangGraph even if langgraph_node is missing
        langgraph_keys = ["langgraph_step", "langgraph_triggers", "langgraph_path", "langgraph_checkpoint_ns"]
        if any(key in metadata for key in langgraph_keys):
            return "node"
        
        # Check if this is a LangGraph graph by name
        # LangGraph sets name="LangGraph" for graph executions
        if name and name.lower() in ("langgraph", "pregel", "compiledstategraph"):
            return "graph"
        
        # Check tags for LangGraph markers (e.g., "graph:step:1")
        if any(tag.startswith("graph:") for tag in tags):
            return "node"
        
        # Fallback: check serialized data if available (for older LangGraph versions)
        if serialized:
            id_list = serialized.get("id", [])
            serialized_name = serialized.get("name", "")
            
            if "Pregel" in str(id_list) or "CompiledStateGraph" in str(id_list):
                return "graph"
            elif "StateGraph" in str(id_list):
                return "graph"
            elif any(x in serialized_name.lower() for x in ["pregel", "graph"]):
                return "graph"
            elif "langgraph" in str(serialized).lower():
                return "node"
        
        # This is not a LangGraph-specific chain
        return "other"

    def _extract_chain_name_from_metadata(
        self,
        name: Optional[str],
        metadata: Optional[Dict[str, Any]],
        serialized: Optional[Dict[str, Any]],
    ) -> str:
        """Extract chain/node name from metadata or name parameter.
        
        LangGraph provides:
        - name parameter: "LangGraph" for graph, node name for nodes
        - metadata["langgraph_node"]: node name for node executions
        """
        metadata = metadata or {}
        
        # For nodes, prefer langgraph_node from metadata
        if "langgraph_node" in metadata:
            return metadata["langgraph_node"]
        
        # Use name parameter if available
        if name:
            return name
        
        # Fallback to serialized data
        if serialized:
            serialized_name = serialized.get("name")
            if serialized_name:
                return serialized_name
            
            id_list = serialized.get("id")
            if id_list and isinstance(id_list, list) and len(id_list) > 0:
                return id_list[-1]
        
        return "unknown"

    def _get_chain_type(self, serialized: Optional[Dict[str, Any]]) -> str:
        """Legacy method for backwards compatibility."""
        return self._get_chain_type_from_metadata(None, None, None, serialized)

    def _extract_chain_name(self, serialized: Optional[Dict[str, Any]]) -> str:
        """Legacy method for backwards compatibility."""
        return self._extract_chain_name_from_metadata(None, None, serialized)

    def _extract_llm_name(self, serialized: Optional[Dict[str, Any]]) -> str:
        """Extract LLM model name from serialized data."""
        if not serialized:
            return "unknown"
        
        kwargs = serialized.get("kwargs", {})
        if kwargs:
            model = kwargs.get("model_name") or kwargs.get("model") or kwargs.get("deployment_name")
            if model:
                return model

        return self._extract_chain_name(serialized)

    # =========================================================================
    # Parallel Execution Detection (Send API)
    # =========================================================================

    def _is_send_object(self, obj: Any) -> bool:
        """Check if obj is a LangGraph Send object."""
        if obj is None:
            return False
        
        # Check by type name to avoid import issues
        type_name = type(obj).__name__
        if type_name == "Send":
            return True
        
        # Also check module path for langgraph.types.Send
        module = getattr(type(obj), "__module__", "")
        if "langgraph" in module and type_name == "Send":
            return True
        
        return False

    def _find_send_objects(self, outputs: Any) -> List[Any]:
        """Find all Send objects in conditional edge output.
        
        When a conditional edge returns Send objects, it triggers parallel execution.
        """
        sends = []
        
        if self._is_send_object(outputs):
            sends.append(outputs)
        elif isinstance(outputs, (list, tuple)):
            for item in outputs:
                if self._is_send_object(item):
                    sends.append(item)
        
        return sends

    def _maybe_emit_parallel_fanout(
        self,
        run: "ArzuleRun",
        from_node: str,
        outputs: Any,
        span_id: str,
        parent_span: Optional[str],
    ) -> None:
        """Emit parallel.fanout event if output contains Send objects."""
        sends = self._find_send_objects(outputs)
        
        if not sends:
            return
        
        # Extract target nodes and count
        target_nodes = []
        for send in sends:
            node = getattr(send, "node", None)
            if node:
                target_nodes.append(node)
        
        if not target_nodes:
            return
        
        # Emit parallel fanout event
        evt = evt_parallel_fanout(
            run=run,
            span_id=new_span_id(),
            parent_span_id=parent_span or run.current_parent_span_id(),
            source_node=from_node,
            target_nodes=list(set(target_nodes)),  # Unique nodes
            send_count=len(sends),
        )
        run.emit(evt)

    # =========================================================================
    # Handoff Detection (for multi-agent coordination)
    # =========================================================================

    def _is_command_object(self, obj: Any) -> bool:
        """Check if obj is a LangGraph Command object."""
        if obj is None:
            return False
        
        # Check by type name to avoid import issues
        type_name = type(obj).__name__
        if type_name == "Command":
            return True
        
        # Also check module path for langgraph.types.Command
        module = getattr(type(obj), "__module__", "")
        if "langgraph" in module and type_name == "Command":
            return True
        
        return False

    def _extract_command_goto(self, obj: Any) -> Optional[str]:
        """Extract goto destination from a Command object."""
        if not self._is_command_object(obj):
            return None
        
        goto = getattr(obj, "goto", None)
        if goto is None:
            return None
        
        # goto can be a string, Send object, or sequence
        if isinstance(goto, str):
            return goto
        
        # Handle Send object
        if hasattr(goto, "node"):
            return goto.node
        
        # Handle sequence (take first)
        if isinstance(goto, (list, tuple)) and len(goto) > 0:
            first = goto[0]
            if isinstance(first, str):
                return first
            if hasattr(first, "node"):
                return first.node
        
        return None

    def _extract_command_update(self, obj: Any) -> Any:
        """Extract update payload from a Command object."""
        if not self._is_command_object(obj):
            return None
        return getattr(obj, "update", None)

    def _find_commands_in_output(self, outputs: Any) -> List[Any]:
        """Find all Command objects in the output.
        
        Outputs can be:
        - A single Command object
        - A dict containing Command objects
        - A list containing Command objects
        """
        commands = []
        
        if self._is_command_object(outputs):
            commands.append(outputs)
        elif isinstance(outputs, (list, tuple)):
            for item in outputs:
                if self._is_command_object(item):
                    commands.append(item)
        elif isinstance(outputs, dict):
            for value in outputs.values():
                if self._is_command_object(value):
                    commands.append(value)
        
        return commands

    def _extract_state_routing_target(self, outputs: Any) -> Optional[str]:
        """Extract routing target from state-based output.
        
        LangGraph commonly uses state-based routing where nodes return:
        - {"next_agent": "target"}
        - {"next": "target"}
        - {"goto": "target"}
        - {"route": "target"}
        
        Returns the target node name, or None if not found.
        """
        if not isinstance(outputs, dict):
            return None
        
        # Common routing field names in LangGraph patterns
        routing_fields = ["next_agent", "next", "goto", "route", "router", "target"]
        
        for field in routing_fields:
            if field in outputs:
                value = outputs[field]
                if isinstance(value, str) and value:
                    # Skip end markers
                    if value.lower() in ("end", "__end__", "none"):
                        return None
                    return value
        
        return None

    def _maybe_emit_handoff_proposed(
        self,
        run: "ArzuleRun",
        from_node: str,
        outputs: Any,
        span_id: str,
        parent_span: Optional[str],
    ) -> None:
        """Emit handoff.proposed event if output contains a Command with goto or state routing."""
        # Strategy 1: Detect Command-based routing (e.g., Command(goto="target"))
        commands = self._find_commands_in_output(outputs)
        
        for cmd in commands:
            to_node = self._extract_command_goto(cmd)
            if not to_node:
                continue
            
            # Skip END marker
            if to_node == "__end__":
                continue
            
            self._emit_handoff_proposed(
                run=run,
                from_node=from_node,
                to_node=to_node,
                payload=self._extract_command_update(cmd),
                handoff_type="langgraph_command",
                parent_span=parent_span,
            )
        
        # Strategy 2: Detect state-based routing (e.g., {"next_agent": "target"})
        state_target = self._extract_state_routing_target(outputs)
        if state_target:
            # Don't emit duplicate handoffs for same target
            # (Command detection might have already picked this up)
            existing_targets = set()
            for key, pending in self._handoff_pending.items():
                if pending["from_node"] == from_node:
                    existing_targets.add(pending["to_node"])
            
            if state_target not in existing_targets:
                # Extract meaningful context for the handoff payload
                # This is what the target node will work on
                payload = self._build_handoff_proposed_payload(outputs, from_node, state_target)
                
                self._emit_handoff_proposed(
                    run=run,
                    from_node=from_node,
                    to_node=state_target,
                    payload=payload,
                    handoff_type="langgraph_state",
                    parent_span=parent_span,
                )

    def _build_handoff_proposed_payload(
        self,
        outputs: Dict[str, Any],
        from_node: str,
        to_node: str,
    ) -> Dict[str, Any]:
        """Build a meaningful payload for handoff.proposed events.
        
        For LangGraph state-based routing, the "task" being handed off is the
        accumulated state that the target node will receive. This includes:
        - messages: The conversation/query history
        - task_results: Results from prior agents (context for the target)
        - Any other relevant state fields
        
        This payload will be used by the semantic analyzer to compare with the
        target node's output for drift detection.
        """
        payload = {
            "routing": f"{from_node} -> {to_node}",
        }
        
        # Extract messages (the core input for most LangGraph agents)
        if "messages" in outputs:
            messages_content = self._extract_message_content(outputs["messages"])
            if messages_content:
                # Join list of messages into a single string
                if isinstance(messages_content, list):
                    payload["task"] = "\n".join(str(m) for m in messages_content)
                else:
                    payload["task"] = messages_content
        
        # Extract task_results (accumulated context from prior agents)
        if "task_results" in outputs:
            task_results = outputs["task_results"]
            if isinstance(task_results, dict) and task_results:
                context_parts = []
                for agent, result in task_results.items():
                    content = self._extract_message_content(result)
                    if content:
                        context_parts.append(f"[{agent}]: {content[:500]}")
                if context_parts:
                    payload["context"] = "\n".join(context_parts)
        
        # Extract any output/result content from the routing node
        for field in ["output", "result", "content", "response"]:
            if field in outputs:
                content = self._extract_message_content(outputs[field])
                if content and "task" not in payload:
                    payload["task"] = content
        
        return payload

    def _emit_handoff_proposed(
        self,
        run: "ArzuleRun",
        from_node: str,
        to_node: str,
        payload: Any,
        handoff_type: str,
        parent_span: Optional[str],
    ) -> None:
        """Internal method to emit handoff.proposed and register tracking."""
        # Generate handoff key
        handoff_key = str(uuid_module.uuid4())
        
        # Store pending handoff for ack/complete correlation
        self._handoff_pending[handoff_key] = {
            "from_node": from_node,
            "to_node": to_node,
            "handoff_type": handoff_type,
        }
        
        # Track that this node is a handoff target
        if to_node not in self._handoff_targets:
            self._handoff_targets[to_node] = []
        self._handoff_targets[to_node].append(handoff_key)
        
        # Emit handoff.proposed event
        evt = evt_handoff_proposed(
            run=run,
            span_id=new_span_id(),
            parent_span_id=parent_span or run.current_parent_span_id(),
            handoff_key=handoff_key,
            from_node=from_node,
            to_node=to_node,
            payload=payload,
            handoff_type=handoff_type,
        )
        run.emit(evt)

    def _maybe_emit_implicit_handoff_proposed(
        self,
        run: "ArzuleRun",
        to_node: str,
        inputs: Any,
        metadata: Optional[Dict[str, Any]],
        span_id: str,
        parent_span: Optional[str],
    ) -> None:
        """Detect and emit handoff.proposed for implicit handoffs via conditional edges.
        
        LangGraph's conditional edges (add_conditional_edges) route based on a router
        function, not based on explicit routing fields in node output. This means the
        source node doesn't know (or signal) where execution will go next.
        
        We detect these implicit handoffs by examining langgraph_triggers metadata
        when a node STARTS. The triggers tell us which node(s) caused this execution.
        
        Example metadata:
            {"langgraph_triggers": ["intake"], "langgraph_step": 1, ...}
        
        This indicates that "intake" node triggered the current node via conditional routing.
        We emit handoff.proposed for this transition so the semantic analyzer can track it.
        """
        if not metadata:
            return
        
        triggers = metadata.get("langgraph_triggers")
        if not triggers:
            return
        
        # Normalize triggers to list
        if not isinstance(triggers, (list, tuple)):
            triggers = [triggers]
        
        # Skip if this looks like a start trigger (not a handoff)
        # __start__ is the entry point, not a real node handoff
        skip_triggers = {"__start__", "start", "__end__", "end"}
        
        for trigger in triggers:
            trigger_str = str(trigger)
            
            # Skip internal/entry triggers
            if trigger_str.lower() in skip_triggers:
                continue
            
            # Skip if we already have a pending handoff for this exact transition
            # (source node may have explicitly set next_agent)
            already_pending = False
            for key, pending in self._handoff_pending.items():
                if pending.get("from_node") == trigger_str and pending.get("to_node") == to_node:
                    already_pending = True
                    break
            
            if already_pending:
                continue
            
            # Build payload from inputs (the state being passed to this node)
            payload = self._build_implicit_handoff_payload(inputs, trigger_str, to_node)
            
            # Emit handoff.proposed for this implicit transition
            self._emit_handoff_proposed(
                run=run,
                from_node=trigger_str,
                to_node=to_node,
                payload=payload,
                handoff_type="langgraph_conditional",
                parent_span=parent_span,
            )

    def _build_implicit_handoff_payload(
        self,
        inputs: Any,
        from_node: str,
        to_node: str,
    ) -> Dict[str, Any]:
        """Build payload for implicit handoff from conditional edge routing.
        
        For implicit handoffs, the payload is the state passed to the target node.
        This may include messages, accumulated results, etc.
        """
        payload = {
            "routing": f"{from_node} -> {to_node}",
            "handoff_type": "conditional_edge",
        }
        
        if not inputs or not isinstance(inputs, dict):
            return payload
        
        # Extract meaningful content from inputs
        # Similar to _build_handoff_proposed_payload but working with input state
        
        # Messages (most common LangGraph pattern)
        if "messages" in inputs:
            messages_content = self._extract_message_content(inputs["messages"])
            if messages_content:
                if isinstance(messages_content, list):
                    payload["task"] = "\n".join(str(m) for m in messages_content[-5:])  # Last 5 messages
                else:
                    payload["task"] = str(messages_content)[:2000]
        
        # Task results / context from prior agents
        if "task_results" in inputs:
            task_results = inputs["task_results"]
            if isinstance(task_results, dict) and task_results:
                context_parts = []
                for agent, result in list(task_results.items())[:5]:  # Limit to 5 agents
                    content = self._extract_message_content(result)
                    if content:
                        context_parts.append(f"[{agent}]: {str(content)[:500]}")
                if context_parts:
                    payload["context"] = "\n".join(context_parts)
        
        # Branch results (for pipeline patterns like scenario_async_complex)
        if "branch_results" in inputs:
            branch_results = inputs["branch_results"]
            if isinstance(branch_results, list) and branch_results:
                branch_summaries = []
                for br in branch_results[:5]:  # Limit to 5 branches
                    if isinstance(br, dict):
                        name = br.get("branch_name", "unknown")
                        content = br.get("content", "")[:300]
                        branch_summaries.append(f"[{name}]: {content}")
                if branch_summaries:
                    payload["branch_context"] = "\n".join(branch_summaries)
        
        # Aggregated content
        if "aggregated_content" in inputs:
            content = inputs["aggregated_content"]
            if content:
                payload["aggregated_content"] = str(content)[:2000]
        
        # Original query (useful context)
        if "original_query" in inputs:
            payload["original_query"] = str(inputs["original_query"])[:500]
        
        return payload

    def _maybe_emit_handoff_ack(
        self,
        run: "ArzuleRun",
        node_name: str,
        span_id: str,
        parent_span: Optional[str],
    ) -> None:
        """Emit handoff.ack event if this node is a handoff target."""
        handoff_keys = self._handoff_targets.get(node_name, [])
        
        for handoff_key in handoff_keys:
            pending = self._handoff_pending.get(handoff_key)
            if not pending:
                continue
            
            from_node = pending.get("from_node", "unknown")
            
            evt = evt_handoff_ack(
                run=run,
                span_id=new_span_id(),
                parent_span_id=parent_span or run.current_parent_span_id(),
                handoff_key=handoff_key,
                from_node=from_node,
                to_node=node_name,
            )
            run.emit(evt)

    def _extract_meaningful_payload(self, outputs: Any) -> Any:
        """Extract meaningful content from LangGraph node outputs.
        
        LangGraph nodes typically return dicts like:
        {
            "messages": [AIMessage(...)],  # Actual LLM responses
            "task_results": {...},          # Accumulated work results
            "current_agent": "...",         # Metadata
            "next_agent": "...",            # Routing (exclude this)
        }
        
        We want to extract the meaningful content for drift analysis,
        filtering out routing/metadata fields. The extracted payload
        uses standardized field names that the semantic analyzer expects:
        - "result" for main output content
        - "output" for additional content
        """
        if outputs is None:
            return None
        
        if not isinstance(outputs, dict):
            return {"result": str(outputs)}
        
        # Fields to exclude (routing and internal metadata)
        exclude_fields = {
            "next_agent", "next", "goto", "route", "router", "target",
            "current_agent", "__interrupt__",
        }
        
        extracted = {}
        
        # Priority 1: Extract from "messages" field (most common in LangGraph)
        # This is where the actual LLM response lives
        if "messages" in outputs:
            messages_content = self._extract_message_content(outputs["messages"])
            if messages_content:
                extracted["result"] = messages_content
        
        # Priority 2: Extract from "task_results" (accumulated work output)
        if "task_results" in outputs:
            task_results = outputs["task_results"]
            if isinstance(task_results, dict):
                # Get the most recent result (last added key)
                # Or combine all results
                results_text = []
                for agent, result in task_results.items():
                    content = self._extract_message_content(result)
                    if content:
                        results_text.append(f"[{agent}]: {content}")
                if results_text:
                    extracted["task_results"] = "\n".join(results_text)
        
        # Priority 3: Direct output/result fields
        for field in ["output", "result", "content", "response"]:
            if field in outputs and field not in extracted:
                value = self._extract_message_content(outputs[field])
                if value:
                    extracted[field] = value
        
        # If we found meaningful content, check it's not just a terminal marker
        if extracted:
            # Filter out terminal-only payloads
            if not self._is_terminal_payload(extracted):
                return extracted
        
        # Fallback: return all non-routing fields
        result = {}
        for key, value in outputs.items():
            if key not in exclude_fields and not key.startswith("_"):
                content = self._extract_message_content(value)
                if content and not self._is_terminal_value(content):
                    result[key] = content
        
        # If result is empty or terminal-only, return None to skip drift analysis
        if not result or self._is_terminal_payload(result):
            return None
        
        return result

    def _is_terminal_value(self, value: Any) -> bool:
        """Check if a value is a terminal/routing marker that shouldn't be analyzed."""
        if value is None:
            return True
        
        if isinstance(value, str):
            normalized = value.lower().strip()
            return normalized in ("end", "__end__", "none", "null", "")
        
        if isinstance(value, dict):
            # Check for nested terminal markers like {"result": "end"}
            if len(value) == 1:
                only_value = list(value.values())[0]
                return self._is_terminal_value(only_value)
        
        return False

    def _is_terminal_payload(self, payload: Dict[str, Any]) -> bool:
        """Check if an entire payload is just terminal markers."""
        if not payload:
            return True
        
        # If all values are terminal, the payload is terminal
        for key, value in payload.items():
            if not self._is_terminal_value(value):
                return False
        
        return True

    def _extract_message_content(self, value: Any) -> Any:
        """Extract text content from LangChain message objects.
        
        Handles:
        - Single message objects (AIMessage, HumanMessage, etc.)
        - Lists of messages
        - Nested structures
        """
        if value is None:
            return None
        
        # Check if it's a LangChain message object
        if hasattr(value, "content"):
            return value.content
        
        # Handle list of messages
        if isinstance(value, (list, tuple)):
            extracted = []
            for item in value:
                if hasattr(item, "content"):
                    extracted.append(item.content)
                else:
                    extracted.append(item)
            # If it's a single item list, return the item directly
            if len(extracted) == 1:
                return extracted[0]
            return extracted
        
        # Handle dict recursively
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                result[k] = self._extract_message_content(v)
            return result
        
        return value

    def _maybe_emit_handoff_complete(
        self,
        run: "ArzuleRun",
        node_name: str,
        result: Any,
        span_id: str,
        parent_span: Optional[str],
        error: Optional[BaseException] = None,
    ) -> None:
        """Emit handoff.complete event if this node was a handoff target."""
        handoff_keys = self._handoff_targets.pop(node_name, [])
        
        for handoff_key in handoff_keys:
            pending = self._handoff_pending.pop(handoff_key, None)
            if not pending:
                continue
            
            from_node = pending.get("from_node", "unknown")
            status = "error" if error else "ok"
            
            # Extract meaningful payload (filter out routing fields and terminal markers)
            meaningful_result = self._extract_meaningful_payload(result) if not error else None
            
            # Skip emitting handoff.complete if the result is just a terminal marker
            # This prevents false positive drift detection for final/coordinator nodes
            if meaningful_result is None and not error:
                # Still track that we completed, but mark as terminal
                evt = evt_handoff_complete(
                    run=run,
                    span_id=new_span_id(),
                    parent_span_id=parent_span or run.current_parent_span_id(),
                    handoff_key=handoff_key,
                    from_node=from_node,
                    to_node=node_name,
                    result={"_terminal": True, "_note": "Node completed with terminal routing (END)"},
                    status="ok",
                    error=None,
                )
            else:
                evt = evt_handoff_complete(
                    run=run,
                    span_id=new_span_id(),
                    parent_span_id=parent_span or run.current_parent_span_id(),
                    handoff_key=handoff_key,
                    from_node=from_node,
                    to_node=node_name,
                    result=meaningful_result,
                    status=status,
                    error=error,
                )
            run.emit(evt)

