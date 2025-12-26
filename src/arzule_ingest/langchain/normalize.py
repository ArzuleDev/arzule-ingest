"""Normalize LangChain events to Arzule TraceEvent format."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..ids import new_span_id
from ..sanitize import sanitize, truncate_string

if TYPE_CHECKING:
    from ..run import ArzuleRun


def _safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get attribute from object."""
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def _extract_chain_name(serialized: Optional[Dict[str, Any]]) -> str:
    """Extract chain/runnable name from serialized data."""
    if not serialized:
        return "unknown"

    # Try different fields that might contain the name
    name = serialized.get("name")
    if name:
        return name

    # Try id field (list of class names)
    id_list = serialized.get("id")
    if id_list and isinstance(id_list, list) and len(id_list) > 0:
        return id_list[-1]  # Last element is usually the class name

    # Fallback to repr
    repr_str = serialized.get("repr")
    if repr_str:
        return truncate_string(repr_str, 50)

    return "unknown"


def _extract_llm_name(serialized: Optional[Dict[str, Any]]) -> str:
    """Extract LLM model name from serialized data."""
    if not serialized:
        return "unknown"

    # Try to get model name from kwargs
    kwargs = serialized.get("kwargs", {})
    if kwargs:
        model = kwargs.get("model_name") or kwargs.get("model") or kwargs.get("deployment_name")
        if model:
            return model

    # Fall back to chain name extraction
    return _extract_chain_name(serialized)


def _extract_tool_name(serialized: Optional[Dict[str, Any]]) -> str:
    """Extract tool name from serialized data."""
    if not serialized:
        return "unknown"

    name = serialized.get("name")
    if name:
        return name

    return _extract_chain_name(serialized)


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
# LLM Events
# =============================================================================


def evt_llm_start(
    run: "ArzuleRun",
    span_id: str,
    parent_span_id: Optional[str],
    serialized: Optional[Dict[str, Any]],
    prompts: Optional[List[str]],
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create event for LLM call start."""
    model_name = _extract_llm_name(serialized)
    prompts = prompts or []

    # Truncate prompts for payload
    truncated_prompts = [truncate_string(p, 1000) for p in prompts[:5]]
    if len(prompts) > 5:
        truncated_prompts.append(f"... and {len(prompts) - 5} more prompts")

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "llm.call.start",
        "status": "ok",
        "summary": f"llm call: {model_name}",
        "attrs_compact": {
            "model": model_name,
            "prompt_count": len(prompts),
            "tags": tags,
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
    response: Any,
) -> Dict[str, Any]:
    """Create event for LLM call end."""
    # Extract response content
    generations = []
    token_usage = {}

    if hasattr(response, "generations"):
        for gen_list in response.generations[:3]:
            if isinstance(gen_list, list):
                for gen in gen_list[:2]:
                    text = _safe_getattr(gen, "text", str(gen))
                    generations.append(truncate_string(text, 500))
            else:
                text = _safe_getattr(gen_list, "text", str(gen_list))
                generations.append(truncate_string(text, 500))

    if hasattr(response, "llm_output") and response.llm_output:
        token_usage = response.llm_output.get("token_usage", {})

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "llm.call.end",
        "status": "ok",
        "summary": "llm response received",
        "attrs_compact": {
            "generation_count": len(generations),
            "total_tokens": token_usage.get("total_tokens"),
            "prompt_tokens": token_usage.get("prompt_tokens"),
            "completion_tokens": token_usage.get("completion_tokens"),
        },
        "payload": {
            "generations": generations,
            "token_usage": token_usage,
        },
    }


def evt_llm_error(
    run: "ArzuleRun",
    span_id: Optional[str],
    parent_span_id: Optional[str],
    error: BaseException,
) -> Dict[str, Any]:
    """Create event for LLM call error."""
    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "llm.call.error",
        "status": "error",
        "summary": f"llm error: {type(error).__name__}",
        "attrs_compact": {
            "error_type": type(error).__name__,
        },
        "payload": {
            "error": truncate_string(str(error), 500),
        },
    }


# =============================================================================
# Chain Events
# =============================================================================


def evt_chain_start(
    run: "ArzuleRun",
    span_id: str,
    parent_span_id: Optional[str],
    serialized: Optional[Dict[str, Any]],
    inputs: Optional[Dict[str, Any]],
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create event for chain execution start."""
    chain_name = _extract_chain_name(serialized)

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "chain.start",
        "status": "ok",
        "summary": f"chain start: {chain_name}",
        "attrs_compact": {
            "chain_name": chain_name,
            "tags": tags,
        },
        "payload": {
            "inputs": sanitize(inputs) if inputs else {},
            "metadata": sanitize(metadata) if metadata else {},
        },
    }


def evt_chain_end(
    run: "ArzuleRun",
    span_id: Optional[str],
    parent_span_id: Optional[str],
    outputs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create event for chain execution end."""
    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "chain.end",
        "status": "ok",
        "summary": "chain completed",
        "attrs_compact": {},
        "payload": {
            "outputs": sanitize(outputs) if outputs else {},
        },
    }


def evt_chain_error(
    run: "ArzuleRun",
    span_id: Optional[str],
    parent_span_id: Optional[str],
    error: BaseException,
) -> Dict[str, Any]:
    """Create event for chain execution error."""
    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "chain.error",
        "status": "error",
        "summary": f"chain error: {type(error).__name__}",
        "attrs_compact": {
            "error_type": type(error).__name__,
        },
        "payload": {
            "error": truncate_string(str(error), 500),
        },
    }


# =============================================================================
# Tool Events
# =============================================================================


def evt_tool_start(
    run: "ArzuleRun",
    span_id: str,
    parent_span_id: Optional[str],
    serialized: Optional[Dict[str, Any]],
    input_str: Optional[str],
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create event for tool invocation start."""
    tool_name = _extract_tool_name(serialized)
    input_str = input_str or ""

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "tool.call.start",
        "status": "ok",
        "summary": f"tool call: {tool_name}",
        "attrs_compact": {
            "tool_name": tool_name,
            "tags": tags,
        },
        "payload": {
            "tool_input": truncate_string(input_str, 1000),
            "metadata": sanitize(metadata) if metadata else {},
        },
    }


def evt_tool_end(
    run: "ArzuleRun",
    span_id: Optional[str],
    parent_span_id: Optional[str],
    output: Any,
) -> Dict[str, Any]:
    """Create event for tool invocation end."""
    output_str = str(output) if output is not None else ""

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "tool.call.end",
        "status": "ok",
        "summary": "tool completed",
        "attrs_compact": {},
        "payload": {
            "tool_output": truncate_string(output_str, 1000),
        },
    }


def evt_tool_error(
    run: "ArzuleRun",
    span_id: Optional[str],
    parent_span_id: Optional[str],
    error: BaseException,
) -> Dict[str, Any]:
    """Create event for tool invocation error."""
    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "tool.call.error",
        "status": "error",
        "summary": f"tool error: {type(error).__name__}",
        "attrs_compact": {
            "error_type": type(error).__name__,
        },
        "payload": {
            "error": truncate_string(str(error), 500),
        },
    }


# =============================================================================
# Agent Events
# =============================================================================


def _extract_agent_name(
    action_or_finish: Any,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Extract agent name from various sources in LangChain/LangGraph.

    Checks in order of priority:
    1. Metadata fields: agent_name, agent, name, node_name
    2. Tags with 'agent:' prefix
    3. Tool input fields: sender, agent, from_agent
    4. Action log parsing for agent identification patterns
    """
    # 1. Check metadata for agent identifiers
    if metadata:
        for key in ("agent_name", "agent", "name", "node_name", "langgraph_node"):
            val = metadata.get(key)
            if val and isinstance(val, str):
                return val

    # 2. Check tags for agent: prefix pattern (e.g., "agent:researcher")
    if tags:
        for tag in tags:
            if isinstance(tag, str):
                if tag.startswith("agent:"):
                    return tag[6:]  # Remove "agent:" prefix
                if tag.startswith("node:"):
                    return tag[5:]  # Remove "node:" prefix

    # 3. Check tool_input for sender/agent fields (common in multi-agent patterns)
    tool_input = _safe_getattr(action_or_finish, "tool_input", None)
    if isinstance(tool_input, dict):
        for key in ("sender", "agent", "from_agent", "agent_name", "current_agent"):
            val = tool_input.get(key)
            if val and isinstance(val, str):
                return val

    # 4. Check return_values for AgentFinish
    return_values = _safe_getattr(action_or_finish, "return_values", None)
    if isinstance(return_values, dict):
        for key in ("agent", "agent_name", "sender"):
            val = return_values.get(key)
            if val and isinstance(val, str):
                return val

    # 5. Try to extract from log if it contains agent identification
    log = _safe_getattr(action_or_finish, "log", "")
    if log and isinstance(log, str):
        # Common patterns: "Agent: ResearcherAgent" or "[ResearcherAgent]"
        import re
        # Pattern: "Agent: <name>" or "Agent <name>"
        match = re.search(r"Agent[:\s]+([A-Za-z0-9_-]+)", log)
        if match:
            return match.group(1)
        # Pattern: "[AgentName]" at start of log
        match = re.match(r"\[([A-Za-z0-9_-]+)\]", log)
        if match:
            return match.group(1)

    return None


def _build_agent_info(agent_name: Optional[str]) -> Dict[str, Any]:
    """Build agent info dict from agent name."""
    if agent_name:
        return {"id": f"langchain:agent:{agent_name}", "role": agent_name}
    return {"id": "langchain:agent", "role": "agent"}


def evt_agent_action(
    run: "ArzuleRun",
    span_id: str,
    parent_span_id: Optional[str],
    action: Any,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create event for agent action."""
    tool = _safe_getattr(action, "tool", "unknown")
    tool_input = _safe_getattr(action, "tool_input", {})
    log = _safe_getattr(action, "log", "")

    # Extract agent name from action, tags, or metadata
    agent_name = _extract_agent_name(action, tags, metadata)

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": _build_agent_info(agent_name),
        "event_type": "agent.action",
        "status": "ok",
        "summary": f"agent action: {tool}" + (f" ({agent_name})" if agent_name else ""),
        "attrs_compact": {
            "tool": tool,
            "agent_name": agent_name,
            "tags": tags,
        },
        "payload": {
            "tool_input": sanitize(tool_input),
            "log": truncate_string(log, 500),
            "metadata": sanitize(metadata) if metadata else {},
        },
    }


def evt_agent_finish(
    run: "ArzuleRun",
    span_id: Optional[str],
    parent_span_id: Optional[str],
    finish: Any,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create event for agent finish."""
    return_values = _safe_getattr(finish, "return_values", {})
    log = _safe_getattr(finish, "log", "")

    # Extract agent name from finish, tags, or metadata
    agent_name = _extract_agent_name(finish, tags, metadata)

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": _build_agent_info(agent_name),
        "event_type": "agent.finish",
        "status": "ok",
        "summary": "agent finished" + (f" ({agent_name})" if agent_name else ""),
        "attrs_compact": {
            "agent_name": agent_name,
            "tags": tags,
        },
        "payload": {
            "return_values": sanitize(return_values),
            "log": truncate_string(log, 500),
            "metadata": sanitize(metadata) if metadata else {},
        },
    }


# =============================================================================
# Retriever Events
# =============================================================================


def evt_retriever_start(
    run: "ArzuleRun",
    span_id: str,
    parent_span_id: Optional[str],
    serialized: Optional[Dict[str, Any]],
    query: Optional[str],
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create event for retriever start."""
    retriever_name = _extract_chain_name(serialized)
    query = query or ""

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "retriever.start",
        "status": "ok",
        "summary": f"retriever: {retriever_name}",
        "attrs_compact": {
            "retriever_name": retriever_name,
            "tags": tags,
        },
        "payload": {
            "query": truncate_string(query, 500),
            "metadata": sanitize(metadata) if metadata else {},
        },
    }


def evt_retriever_end(
    run: "ArzuleRun",
    span_id: Optional[str],
    parent_span_id: Optional[str],
    documents: Optional[List[Any]],
) -> Dict[str, Any]:
    """Create event for retriever end."""
    documents = documents or []

    # Extract document summaries
    doc_summaries = []
    for doc in documents[:5]:
        page_content = _safe_getattr(doc, "page_content", str(doc))
        doc_metadata = _safe_getattr(doc, "metadata", {})
        doc_summaries.append({
            "content_preview": truncate_string(page_content, 200),
            "metadata": sanitize(doc_metadata),
        })

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "retriever.end",
        "status": "ok",
        "summary": f"retrieved {len(documents)} documents",
        "attrs_compact": {
            "document_count": len(documents),
        },
        "payload": {
            "documents": doc_summaries,
        },
    }


def evt_retriever_error(
    run: "ArzuleRun",
    span_id: Optional[str],
    parent_span_id: Optional[str],
    error: BaseException,
) -> Dict[str, Any]:
    """Create event for retriever error."""
    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "retriever.error",
        "status": "error",
        "summary": f"retriever error: {type(error).__name__}",
        "attrs_compact": {
            "error_type": type(error).__name__,
        },
        "payload": {
            "error": truncate_string(str(error), 500),
        },
    }

