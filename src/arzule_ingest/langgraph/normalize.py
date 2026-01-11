"""Normalize LangGraph events to Arzule TraceEvent format."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..ids import new_span_id
from ..logger import get_logger
from ..sanitize import sanitize, truncate_string

logger = get_logger()

if TYPE_CHECKING:
    from ..run import ArzuleRun


def _safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get attribute from object."""
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def _extract_token_usage(response: Any) -> Optional[Dict[str, int]]:
    """Extract token usage from LLM response.
    
    Supports multiple formats from different LLM providers:
    - OpenAI/LiteLLM: response.usage.{prompt_tokens, completion_tokens, total_tokens}
    - LangChain LLMResult: response.llm_output["token_usage"]
    - usage_metadata: response.usage_metadata.{input_tokens, output_tokens}
    - Direct attributes: response.{prompt_tokens, completion_tokens, total_tokens}
    - Generations with usage: response.generations[0][0].generation_info["usage"]
    
    Returns:
        Dict with prompt_tokens, completion_tokens, total_tokens or None if not available
    """
    if not response:
        return None
    
    result: Dict[str, int] = {}
    
    # Try response.usage (OpenAI/LiteLLM format)
    usage = _safe_getattr(response, "usage", None)
    if usage:
        prompt = _safe_getattr(usage, "prompt_tokens", None)
        completion = _safe_getattr(usage, "completion_tokens", None)
        total = _safe_getattr(usage, "total_tokens", None)
        
        # Also check for input_tokens/output_tokens naming (Anthropic, etc.)
        if prompt is None:
            prompt = _safe_getattr(usage, "input_tokens", None)
        if completion is None:
            completion = _safe_getattr(usage, "output_tokens", None)
        
        if prompt is not None:
            result["prompt_tokens"] = int(prompt)
        if completion is not None:
            result["completion_tokens"] = int(completion)
        if total is not None:
            result["total_tokens"] = int(total)
        elif prompt is not None and completion is not None:
            result["total_tokens"] = int(prompt) + int(completion)
    
    # Try response.llm_output["token_usage"] (LangChain LLMResult format)
    if not result:
        llm_output = _safe_getattr(response, "llm_output", None)
        if llm_output and isinstance(llm_output, dict):
            token_usage = llm_output.get("token_usage", {})
            if token_usage:
                if "prompt_tokens" in token_usage:
                    result["prompt_tokens"] = int(token_usage["prompt_tokens"])
                if "completion_tokens" in token_usage:
                    result["completion_tokens"] = int(token_usage["completion_tokens"])
                if "total_tokens" in token_usage:
                    result["total_tokens"] = int(token_usage["total_tokens"])
    
    # Try response.usage_metadata (some LangChain providers)
    if not result:
        usage_meta = _safe_getattr(response, "usage_metadata", None)
        if usage_meta:
            input_tokens = _safe_getattr(usage_meta, "input_tokens", None)
            output_tokens = _safe_getattr(usage_meta, "output_tokens", None)
            total_tokens = _safe_getattr(usage_meta, "total_tokens", None)
            
            if input_tokens is not None:
                result["prompt_tokens"] = int(input_tokens)
            if output_tokens is not None:
                result["completion_tokens"] = int(output_tokens)
            if total_tokens is not None:
                result["total_tokens"] = int(total_tokens)
            elif input_tokens is not None and output_tokens is not None:
                result["total_tokens"] = int(input_tokens) + int(output_tokens)
    
    # Try generations[0][0].generation_info (some LangChain LLMResults)
    if not result:
        generations = _safe_getattr(response, "generations", None)
        if generations and isinstance(generations, list) and len(generations) > 0:
            first_gen_list = generations[0]
            if isinstance(first_gen_list, list) and len(first_gen_list) > 0:
                first_gen = first_gen_list[0]
                gen_info = _safe_getattr(first_gen, "generation_info", None)
                if gen_info and isinstance(gen_info, dict):
                    usage = gen_info.get("usage", {})
                    if usage:
                        if "prompt_tokens" in usage:
                            result["prompt_tokens"] = int(usage["prompt_tokens"])
                        if "completion_tokens" in usage:
                            result["completion_tokens"] = int(usage["completion_tokens"])
                        if "total_tokens" in usage:
                            result["total_tokens"] = int(usage["total_tokens"])
    
    # Try generations[0][0].message.usage_metadata (ChatGeneration with AIMessage)
    # This is the PRIMARY location for newer LangChain versions
    if not result:
        generations = _safe_getattr(response, "generations", None)
        if generations and isinstance(generations, list) and len(generations) > 0:
            first_gen_list = generations[0]
            if isinstance(first_gen_list, list) and len(first_gen_list) > 0:
                first_gen = first_gen_list[0]
                # Get the message from ChatGeneration
                message = _safe_getattr(first_gen, "message", None)
                if message:
                    # Try message.usage_metadata (preferred in newer LangChain)
                    usage_meta = _safe_getattr(message, "usage_metadata", None)
                    if usage_meta:
                        input_tokens = _safe_getattr(usage_meta, "input_tokens", None)
                        output_tokens = _safe_getattr(usage_meta, "output_tokens", None)
                        total_tokens = _safe_getattr(usage_meta, "total_tokens", None)
                        
                        # Also try dict access for usage_metadata
                        if input_tokens is None and isinstance(usage_meta, dict):
                            input_tokens = usage_meta.get("input_tokens")
                            output_tokens = usage_meta.get("output_tokens")
                            total_tokens = usage_meta.get("total_tokens")
                        
                        if input_tokens is not None:
                            result["prompt_tokens"] = int(input_tokens)
                        if output_tokens is not None:
                            result["completion_tokens"] = int(output_tokens)
                        if total_tokens is not None:
                            result["total_tokens"] = int(total_tokens)
                        elif input_tokens is not None and output_tokens is not None:
                            result["total_tokens"] = int(input_tokens) + int(output_tokens)
    
    # Try generations[0][0].message.response_metadata["token_usage"] (alternate location)
    if not result:
        generations = _safe_getattr(response, "generations", None)
        if generations and isinstance(generations, list) and len(generations) > 0:
            first_gen_list = generations[0]
            if isinstance(first_gen_list, list) and len(first_gen_list) > 0:
                first_gen = first_gen_list[0]
                message = _safe_getattr(first_gen, "message", None)
                if message:
                    response_meta = _safe_getattr(message, "response_metadata", None)
                    if response_meta and isinstance(response_meta, dict):
                        token_usage = response_meta.get("token_usage", {})
                        if token_usage:
                            if "prompt_tokens" in token_usage:
                                result["prompt_tokens"] = int(token_usage["prompt_tokens"])
                            if "completion_tokens" in token_usage:
                                result["completion_tokens"] = int(token_usage["completion_tokens"])
                            if "total_tokens" in token_usage:
                                result["total_tokens"] = int(token_usage["total_tokens"])
    
    # Try dict-style access (some responses are dicts)
    if not result and isinstance(response, dict):
        usage = response.get("usage", {})
        if usage:
            if "prompt_tokens" in usage:
                result["prompt_tokens"] = int(usage["prompt_tokens"])
            if "completion_tokens" in usage:
                result["completion_tokens"] = int(usage["completion_tokens"])
            if "total_tokens" in usage:
                result["total_tokens"] = int(usage["total_tokens"])

    # Debug logging for token extraction
    if result:
        logger.debug(f"[arzule] LangGraph token usage extracted: {result}")
    else:
        logger.debug(f"[arzule] LangGraph no token usage found. Response type: {type(response).__name__}")

    return result if result else None


def _base(
    run: "ArzuleRun",
    *,
    span_id: Optional[str],
    parent_span_id: Optional[str],
) -> Dict[str, Any]:
    """Build base event fields."""
    return {
        "schema_version": "trace_event.v0_1",
        "run_id": run.run_id,
        "tenant_id": run.tenant_id,
        "project_id": run.project_id,
        "trace_id": run.trace_id,
        "span_id": span_id or new_span_id(),
        "parent_span_id": parent_span_id,
        "seq": run.next_seq(),
        "ts": run.now(),
        "workstream_id": None,
        "task_id": None,
        "raw_ref": {"storage": "inline"},
    }


# =============================================================================
# Graph Events
# =============================================================================


def evt_graph_start(
    run: "ArzuleRun",
    span_id: str,
    parent_span_id: Optional[str],
    graph_name: str,
    input_data: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create event for graph execution start."""
    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "graph.start",
        "status": "ok",
        "summary": f"graph execution: {graph_name}",
        "attrs_compact": {
            "graph_name": graph_name,
            "metadata": metadata,
        },
        "payload": {
            "input": sanitize(input_data),
        },
    }


def evt_graph_end(
    run: "ArzuleRun",
    span_id: Optional[str],
    parent_span_id: Optional[str],
    graph_name: str,
    output_data: Any,
    error: Optional[Exception] = None,
) -> Dict[str, Any]:
    """Create event for graph execution end."""
    status = "error" if error else "ok"
    summary = f"graph completed: {graph_name}"
    if error:
        summary = f"graph error: {graph_name} - {str(error)[:100]}"

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "graph.end",
        "status": status,
        "summary": summary,
        "attrs_compact": {
            "graph_name": graph_name,
            "error": str(error) if error else None,
        },
        "payload": {
            "output": sanitize(output_data) if not error else None,
            "error": str(error) if error else None,
        },
    }


# =============================================================================
# Node Events
# =============================================================================


def _extract_parallel_execution_info(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract parallel execution metadata from LangGraph node metadata.
    
    LangGraph provides useful metadata for understanding parallel execution:
    - langgraph_step: Step number in the graph execution (parallel nodes share same step)
    - langgraph_triggers: List of edges/nodes that triggered this node
    - langgraph_path: Execution path within the graph
    - langgraph_checkpoint_ns: Checkpoint namespace for this execution branch
    
    Returns:
        Dict with extracted parallel execution info
    """
    if not metadata:
        return {}
    
    info: Dict[str, Any] = {}
    
    # Step number - parallel nodes in same superstep share this value
    if "langgraph_step" in metadata:
        info["step"] = metadata["langgraph_step"]
    
    # Triggers - what caused this node to execute (useful for tracking fan-out sources)
    # Filter out internal routing channel triggers (branch:to:*, join:*, split:*)
    if "langgraph_triggers" in metadata:
        triggers = metadata["langgraph_triggers"]
        if isinstance(triggers, (list, tuple)):
            # Filter out internal routing channels from triggers
            filtered_triggers = []
            for t in triggers:
                t_str = str(t).lower()
                if not (t_str.startswith("branch:to:") or t_str.startswith("join:") or t_str.startswith("split:")):
                    filtered_triggers.append(t)
            if filtered_triggers:
                info["triggers"] = filtered_triggers
        else:
            t_str = str(triggers).lower()
            if not (t_str.startswith("branch:to:") or t_str.startswith("join:") or t_str.startswith("split:")):
                info["triggers"] = [str(triggers)]
    
    # Execution path - shows position in graph topology
    if "langgraph_path" in metadata:
        path = metadata["langgraph_path"]
        if isinstance(path, (list, tuple)):
            info["path"] = list(path)
    
    # Checkpoint namespace - unique per execution branch (helps identify parallel branches)
    if "langgraph_checkpoint_ns" in metadata:
        info["checkpoint_ns"] = metadata["langgraph_checkpoint_ns"]
    
    # Task ID for Send API - each Send creates a unique task
    if "langgraph_task_id" in metadata:
        info["task_id"] = metadata["langgraph_task_id"]
        info["is_parallel_task"] = True  # Flag that this is a Send-based parallel task
    
    # Detect if this is likely a parallel execution based on triggers
    # Multiple triggers or triggers containing "fork" patterns indicate parallelism
    if info.get("triggers") and len(info.get("triggers", [])) > 0:
        # Check if triggered by a Send (map-reduce pattern)
        for trigger in info.get("triggers", []):
            if isinstance(trigger, str) and ":send:" in trigger.lower():
                info["is_parallel_task"] = True
                break
    
    return info


def _is_internal_routing_channel(node_name: str) -> bool:
    """Check if this is a LangGraph internal routing channel that should be skipped."""
    if not node_name:
        return False
    lower = node_name.lower()
    return (
        lower.startswith("branch:to:")
        or lower.startswith("join:")
        or lower.startswith("split:")
    )


def evt_node_start(
    run: "ArzuleRun",
    span_id: str,
    parent_span_id: Optional[str],
    node_name: str,
    input_data: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Create event for node execution start.
    
    Returns None if this is an internal routing channel that should be skipped.
    """
    # Skip internal routing channels
    if _is_internal_routing_channel(node_name):
        return None
    
    # Extract parallel execution info from LangGraph metadata
    parallel_info = _extract_parallel_execution_info(metadata)
    
    attrs: Dict[str, Any] = {
        "node_name": node_name,
    }
    
    # Add parallel execution attributes for better tracking
    if parallel_info:
        if "step" in parallel_info:
            attrs["langgraph_step"] = parallel_info["step"]
        if "triggers" in parallel_info:
            attrs["langgraph_triggers"] = parallel_info["triggers"]
        if "task_id" in parallel_info:
            attrs["langgraph_task_id"] = parallel_info["task_id"]
        if parallel_info.get("is_parallel_task"):
            attrs["is_parallel_task"] = True
        if "checkpoint_ns" in parallel_info:
            attrs["checkpoint_ns"] = parallel_info["checkpoint_ns"]
    
    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        # Agent must be an AgentInfo object with id and role for backend to extract agent_role
        "agent": {
            "id": f"langgraph:node:{node_name}",
            "role": node_name,
        },
        "event_type": "agent.start",
        "status": "ok",
        "summary": f"node start: {node_name}",
        "attrs_compact": attrs,
        "payload": {
            "input": sanitize(input_data),
            "metadata": metadata,  # Keep full metadata in payload for reference
        },
    }


def evt_node_end(
    run: "ArzuleRun",
    span_id: Optional[str],
    parent_span_id: Optional[str],
    node_name: str,
    output_data: Any,
    error: Optional[Exception] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Create event for node execution end.
    
    Returns None if this is an internal routing channel that should be skipped.
    """
    # Skip internal routing channels
    if _is_internal_routing_channel(node_name):
        return None
    
    status = "error" if error else "ok"
    summary = f"node completed: {node_name}"
    if error:
        summary = f"node error: {node_name} - {str(error)[:100]}"

    # Extract parallel execution info
    parallel_info = _extract_parallel_execution_info(metadata)
    
    attrs: Dict[str, Any] = {
        "node_name": node_name,
        "error": str(error) if error else None,
    }
    
    # Add parallel execution attributes
    if parallel_info:
        if "step" in parallel_info:
            attrs["langgraph_step"] = parallel_info["step"]
        if parallel_info.get("is_parallel_task"):
            attrs["is_parallel_task"] = True

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        # Agent must be an AgentInfo object with id and role for backend to extract agent_role
        "agent": {
            "id": f"langgraph:node:{node_name}",
            "role": node_name,
        },
        "event_type": "agent.end",
        "status": status,
        "summary": summary,
        "attrs_compact": attrs,
        "payload": {
            "output": sanitize(output_data) if not error else None,
            "error": str(error) if error else None,
        },
    }


# =============================================================================
# Checkpoint Events
# =============================================================================


def evt_checkpoint_save(
    run: "ArzuleRun",
    span_id: str,
    parent_span_id: Optional[str],
    checkpoint_id: str,
    state: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create event for checkpoint save."""
    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "checkpoint.save",
        "status": "ok",
        "summary": f"checkpoint saved: {checkpoint_id[:16]}",
        "attrs_compact": {
            "checkpoint_id": checkpoint_id,
            "metadata": metadata,
        },
        "payload": {
            "state": sanitize(state),
        },
    }


def evt_checkpoint_load(
    run: "ArzuleRun",
    span_id: str,
    parent_span_id: Optional[str],
    checkpoint_id: str,
    state: Any,
) -> Dict[str, Any]:
    """Create event for checkpoint load."""
    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "checkpoint.load",
        "status": "ok",
        "summary": f"checkpoint loaded: {checkpoint_id[:16]}",
        "attrs_compact": {
            "checkpoint_id": checkpoint_id,
        },
        "payload": {
            "state": sanitize(state),
        },
    }


# =============================================================================
# Task Events (for parallel execution tracking)
# =============================================================================


def evt_task_start(
    run: "ArzuleRun",
    span_id: str,
    parent_span_id: Optional[str],
    task_id: str,
    task_name: str,
    task_input: Any,
) -> Dict[str, Any]:
    """Create event for task start (parallel execution)."""
    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "task.start",
        "status": "ok",
        "summary": f"task start: {task_name}",
        "attrs_compact": {
            "task_id": task_id,
            "task_name": task_name,
        },
        "payload": {
            "input": sanitize(task_input),
        },
    }


def evt_task_end(
    run: "ArzuleRun",
    span_id: Optional[str],
    parent_span_id: Optional[str],
    task_id: str,
    task_name: str,
    result: Any,
    error: Optional[Exception] = None,
) -> Dict[str, Any]:
    """Create event for task end."""
    status = "error" if error else "ok"
    summary = f"task completed: {task_name}"
    if error:
        summary = f"task error: {task_name} - {str(error)[:100]}"

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "task.end",
        "status": status,
        "summary": summary,
        "attrs_compact": {
            "task_id": task_id,
            "task_name": task_name,
            "error": str(error) if error else None,
        },
        "payload": {
            "result": sanitize(result) if not error else None,
            "error": str(error) if error else None,
        },
    }


def evt_parallel_fanout(
    run: "ArzuleRun",
    span_id: str,
    parent_span_id: Optional[str],
    source_node: str,
    target_nodes: List[str],
    send_count: int,
) -> Dict[str, Any]:
    """Create event for parallel fan-out (Send API usage).
    
    This is emitted when a conditional edge returns multiple Send objects,
    triggering parallel execution of the target nodes.
    
    Args:
        source_node: The node that initiated the fan-out
        target_nodes: List of unique node names being targeted
        send_count: Total number of Send objects (may be > len(target_nodes) 
                    if same node is targeted multiple times with different inputs)
    """
    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": {
            "id": f"langgraph:node:{source_node}",
            "role": source_node,
        },
        "event_type": "parallel.fanout",
        "status": "ok",
        "summary": f"parallel fan-out: {source_node} -> {len(target_nodes)} nodes ({send_count} tasks)",
        "attrs_compact": {
            "source_node": source_node,
            "target_nodes": target_nodes,
            "send_count": send_count,
            "is_map_reduce": send_count > 1 and len(set(target_nodes)) == 1,
        },
        "payload": {},
    }


def evt_parallel_fanin(
    run: "ArzuleRun",
    span_id: str,
    parent_span_id: Optional[str],
    target_node: str,
    source_nodes: List[str],
    results_count: int,
) -> Dict[str, Any]:
    """Create event for parallel fan-in (results aggregation).
    
    This is emitted when parallel branches complete and their results
    are aggregated via a reducer.
    """
    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": {
            "id": f"langgraph:node:{target_node}",
            "role": target_node,
        },
        "event_type": "parallel.fanin",
        "status": "ok",
        "summary": f"parallel fan-in: {results_count} results -> {target_node}",
        "attrs_compact": {
            "target_node": target_node,
            "source_nodes": source_nodes,
            "results_count": results_count,
        },
        "payload": {},
    }


# =============================================================================
# LLM Events (from nested LangChain calls)
# =============================================================================


def evt_llm_start(
    run: "ArzuleRun",
    span_id: str,
    parent_span_id: Optional[str],
    model_name: str,
    prompts: Optional[List[str]],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create event for LLM call start."""
    prompts = prompts or []

    # Keep full prompts (up to 50k chars each) - sanitized for PII/secrets
    truncated_prompts = [sanitize(truncate_string(p, 50_000)) for p in prompts[:10]]
    if len(prompts) > 10:
        truncated_prompts.append(f"... and {len(prompts) - 10} more prompts")

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "llm.call.start",
        "status": "ok",
        "summary": f"llm call: {model_name}",
        "attrs_compact": {
            "model": model_name,
            "prompt_count": len(prompts),
        },
        "payload": {
            "prompts": truncated_prompts,
            "metadata": sanitize(metadata) if metadata else {},
        },
    }


def evt_llm_end(
    run: "ArzuleRun",
    span_id: Optional[str],
    parent_span_id: Optional[str],
    model_name: str,
    response: Any,
    error: Optional[Exception] = None,
) -> Dict[str, Any]:
    """Create event for LLM call end."""
    status = "error" if error else "ok"
    summary = f"llm completed: {model_name}"
    if error:
        summary = f"llm error: {model_name} - {str(error)[:100]}"

    # Extract token usage using robust multi-format extraction
    token_usage = _extract_token_usage(response) or {}
    
    # Build attrs with flat token fields for consistency with CrewAI tracking
    attrs: Dict[str, Any] = {
        "model": model_name,
        "error": str(error) if error else None,
    }
    
    # Add token counts as flat fields (same format as CrewAI)
    if token_usage:
        if "prompt_tokens" in token_usage:
            attrs["prompt_tokens"] = token_usage["prompt_tokens"]
        if "completion_tokens" in token_usage:
            attrs["completion_tokens"] = token_usage["completion_tokens"]
        if "total_tokens" in token_usage:
            attrs["total_tokens"] = token_usage["total_tokens"]

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "llm.call.end",
        "status": status,
        "summary": summary,
        "attrs_compact": attrs,
        "payload": {
            "response": sanitize(response) if not error else None,
            "token_usage": token_usage if token_usage else None,
            "error": str(error) if error else None,
        },
    }


# =============================================================================
# Tool Events (from nested tool calls)
# =============================================================================


def evt_tool_start(
    run: "ArzuleRun",
    span_id: str,
    parent_span_id: Optional[str],
    tool_name: str,
    tool_input: Any,
) -> Dict[str, Any]:
    """Create event for tool call start."""
    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "tool.call.start",
        "status": "ok",
        "summary": f"tool call: {tool_name}",
        "attrs_compact": {
            "tool": tool_name,
        },
        "payload": {
            "input": sanitize(tool_input),
        },
    }


def evt_tool_end(
    run: "ArzuleRun",
    span_id: Optional[str],
    parent_span_id: Optional[str],
    tool_name: str,
    output: Any,
    error: Optional[Exception] = None,
) -> Dict[str, Any]:
    """Create event for tool call end."""
    status = "error" if error else "ok"
    summary = f"tool completed: {tool_name}"
    if error:
        summary = f"tool error: {tool_name} - {str(error)[:100]}"

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "tool.call.end",
        "status": status,
        "summary": summary,
        "attrs_compact": {
            "tool": tool_name,
            "error": str(error) if error else None,
        },
        "payload": {
            "output": sanitize(output) if not error else None,
            "error": str(error) if error else None,
        },
    }


# =============================================================================
# Handoff Events (for multi-agent coordination)
# =============================================================================


def evt_handoff_proposed(
    run: "ArzuleRun",
    span_id: str,
    parent_span_id: Optional[str],
    handoff_key: str,
    from_node: str,
    to_node: str,
    payload: Any,
    handoff_type: str = "langgraph_command",
) -> Optional[Dict[str, Any]]:
    """Create event when a node routes to another node.
    
    In LangGraph, handoffs occur in three main patterns:
    1. Command-based: Node returns Command(goto="target") - handoff_type="langgraph_command"
    2. State-based: Node sets {"next_agent": "target"} - handoff_type="langgraph_state"
    3. Conditional edge: Router function routes to target - handoff_type="langgraph_conditional"
    
    Pattern 3 is detected by examining langgraph_triggers metadata when a node starts,
    which reveals which node triggered this execution via conditional routing.
    
    Returns None if from_node or to_node are internal routing channels.
    """
    # Skip if from_node or to_node are internal routing channels
    if _is_internal_routing_channel(from_node) or _is_internal_routing_channel(to_node):
        return None
    
    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": {
            "id": f"langgraph:node:{from_node}",
            "role": from_node,
        },
        "event_type": "handoff.proposed",
        "status": "ok",
        "summary": f"handoff: {from_node} -> {to_node}",
        "attrs_compact": {
            "handoff_key": handoff_key,
            "handoff_type": handoff_type,
            "from_agent_role": from_node,
            "to_agent_role": to_node,
            "to_coworker": to_node,  # For compatibility with CrewAI detectors
        },
        "payload": {
            "tool_input": sanitize(payload) if payload else {},
        },
    }


def evt_handoff_ack(
    run: "ArzuleRun",
    span_id: str,
    parent_span_id: Optional[str],
    handoff_key: str,
    from_node: str,
    to_node: str,
) -> Optional[Dict[str, Any]]:
    """Create event when the target node starts (acknowledges the handoff).
    
    In LangGraph, this fires when the destination node begins execution.
    
    Returns None if from_node or to_node are internal routing channels.
    """
    # Skip if from_node or to_node are internal routing channels
    if _is_internal_routing_channel(from_node) or _is_internal_routing_channel(to_node):
        return None
    
    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": {
            "id": f"langgraph:node:{to_node}",
            "role": to_node,
        },
        "event_type": "handoff.ack",
        "status": "ok",
        "summary": f"handoff acknowledged: {to_node} (from {from_node})",
        "attrs_compact": {
            "handoff_key": handoff_key,
            "handoff_type": "langgraph_command",
            "from_agent_role": from_node,
            "to_agent_role": to_node,
        },
        "payload": {},
    }


def evt_handoff_complete(
    run: "ArzuleRun",
    span_id: str,
    parent_span_id: Optional[str],
    handoff_key: str,
    from_node: str,
    to_node: str,
    result: Any,
    status: str = "ok",
    error: Optional[Exception] = None,
) -> Optional[Dict[str, Any]]:
    """Create event when the target node completes (handoff complete).
    
    In LangGraph, this fires when the destination node finishes execution.
    
    Returns None if from_node or to_node are internal routing channels.
    """
    # Skip if from_node or to_node are internal routing channels
    if _is_internal_routing_channel(from_node) or _is_internal_routing_channel(to_node):
        return None
    
    summary = f"handoff complete: {to_node}"
    if error:
        summary = f"handoff failed: {to_node} - {str(error)[:100]}"

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": {
            "id": f"langgraph:node:{to_node}",
            "role": to_node,
        },
        "event_type": "handoff.complete",
        "status": status,
        "summary": summary,
        "attrs_compact": {
            "handoff_key": handoff_key,
            "handoff_type": "langgraph_command",
            "from_agent_role": from_node,
            "to_agent_role": to_node,
            "error": str(error) if error else None,
        },
        "payload": {
            "result": sanitize(result) if result and not error else None,
            "error": str(error) if error else None,
        },
    }








