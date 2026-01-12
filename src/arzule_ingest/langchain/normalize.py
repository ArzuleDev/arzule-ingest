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


def _extract_token_usage(response: Any) -> Dict[str, Any]:
    """Extract token usage from LLM response.
    
    Supports multiple formats from different LLM providers:
    - OpenAI/LiteLLM: response.usage.{prompt_tokens, completion_tokens, total_tokens}
    - LangChain LLMResult: response.llm_output["token_usage"]
    - usage_metadata: response.usage_metadata.{input_tokens, output_tokens}
    - Generations with usage: response.generations[0][0].generation_info["usage"]
    
    Returns:
        Dict with prompt_tokens, completion_tokens, total_tokens (may be empty)
    """
    if not response:
        return {}
    
    result: Dict[str, Any] = {}
    
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
    
    return result


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

    # Keep full prompts (up to 50k chars each) - important for debugging
    # Apply sanitization to redact secrets and PII from user prompts
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
    # Extract response content - keep full generations for debugging
    generations = []

    if hasattr(response, "generations"):
        for gen_list in response.generations[:5]:
            if isinstance(gen_list, list):
                for gen in gen_list[:3]:
                    text = _safe_getattr(gen, "text", str(gen))
                    # Apply sanitization to redact secrets and PII from LLM responses
                    generations.append(sanitize(truncate_string(text, 50_000)))
            else:
                text = _safe_getattr(gen_list, "text", str(gen_list))
                generations.append(sanitize(truncate_string(text, 50_000)))

    # Extract token usage using robust multi-format extraction
    token_usage = _extract_token_usage(response)

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
            "token_usage": token_usage if token_usage else None,
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
            "tool_input": sanitize(truncate_string(input_str, 20_000)),
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
            "tool_output": sanitize(truncate_string(output_str, 20_000)),
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
            "log": sanitize(truncate_string(log, 500)),
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
            "log": sanitize(truncate_string(log, 500)),
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
            "query": sanitize(truncate_string(query, 500)),
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
            "content_preview": sanitize(truncate_string(page_content, 200)),
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

