"""Normalize CrewAI events and hook contexts to TraceEvent format."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Optional

from ..ids import new_span_id
from ..sanitize import sanitize, truncate_string
from .handoff import is_delegation_tool

if TYPE_CHECKING:
    from ..run import ArzuleRun


def _safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get attribute from object."""
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def _compute_input_hash(tool_input: Any) -> Optional[str]:
    """
    Compute a stable hash of tool input for repetition detection.
    
    Returns a 12-character hex hash that can be used to identify
    duplicate tool calls with the same input.
    """
    if tool_input is None:
        return None
    
    try:
        # Serialize with sorted keys for deterministic output
        if isinstance(tool_input, dict):
            serialized = json.dumps(tool_input, sort_keys=True, default=str)
        else:
            serialized = str(tool_input)
        
        return hashlib.md5(serialized.encode()).hexdigest()[:12]
    except Exception:
        return None


def _extract_input_keys(tool_input: Any, max_keys: int = 10) -> list[str]:
    """
    Extract top-level keys from tool input for schema analysis.
    
    Returns sorted list of keys (capped at max_keys) for detecting
    schema drift across tool calls.
    """
    if not isinstance(tool_input, dict):
        return []
    
    try:
        # Get sorted keys, excluding internal arzule metadata
        keys = [k for k in tool_input.keys() if k != "arzule"]
        return sorted(keys)[:max_keys]
    except Exception:
        return []


def _extract_agent_info(agent: Any) -> Optional[dict[str, Any]]:
    """Extract agent info from CrewAI agent object.
    
    Extracts key agent attributes that are useful for trace analysis:
    - role: Agent's function/expertise within the crew
    - goal: Individual objective guiding decision-making
    - allow_delegation: Whether agent can delegate to others (critical for handoff analysis)
    - tools: List of tool names available to the agent
    - max_iter: Maximum iterations before agent must provide answer
    - max_execution_time: Timeout in seconds
    - verbose: Whether detailed logging is enabled
    """
    if not agent:
        return None

    role = _safe_getattr(agent, "role", None) or _safe_getattr(agent, "name", "unknown")
    
    info: dict[str, Any] = {
        "id": f"crewai:role:{role}",
        "role": role,
    }
    
    # Goal - provides context for agent behavior
    goal = _safe_getattr(agent, "goal", None)
    if goal:
        info["goal"] = goal
    
    # Delegation capability - critical for handoff analysis
    allow_delegation = _safe_getattr(agent, "allow_delegation", None)
    if allow_delegation is not None:
        info["allow_delegation"] = bool(allow_delegation)
    
    # Tools available to the agent
    tools = _safe_getattr(agent, "tools", None)
    if tools:
        # Extract tool names, handling both Tool objects and strings
        tool_names = []
        for tool in tools:
            if hasattr(tool, "name"):
                tool_names.append(tool.name)
            elif hasattr(tool, "__name__"):
                tool_names.append(tool.__name__)
            elif isinstance(tool, str):
                tool_names.append(tool)
        if tool_names:
            info["tools"] = tool_names
    
    # Iteration/execution limits
    max_iter = _safe_getattr(agent, "max_iter", None)
    if max_iter is not None:
        info["max_iter"] = max_iter
    
    max_execution_time = _safe_getattr(agent, "max_execution_time", None)
    if max_execution_time is not None:
        info["max_execution_time"] = max_execution_time
    
    # Verbose mode
    verbose = _safe_getattr(agent, "verbose", None)
    if verbose is not None:
        info["verbose"] = bool(verbose)
    
    return info


def extract_agent_info_from_event(event: Any) -> Optional[dict[str, Any]]:
    """Extract agent info dict from a CrewAI event object.
    
    This is a convenience function for the listener to extract agent info
    from event bus events for thread-local agent tracking.
    
    Args:
        event: CrewAI event object (e.g., AgentExecutionStartedEvent)
        
    Returns:
        Agent info dict with 'id' and 'role', or None if no agent
    """
    agent = _safe_getattr(event, "agent", None)
    return _extract_agent_info(agent)


def _extract_task_info(task: Any) -> tuple[Optional[str], Optional[str]]:
    """Extract task ID and description from CrewAI task."""
    if not task:
        return None, None

    task_id = _safe_getattr(task, "id", None) or _safe_getattr(task, "name", None)
    description = _safe_getattr(task, "description", None)
    return task_id, description


def _extract_crew_details(crew: Any) -> dict[str, Any]:
    """Extract detailed crew attributes from CrewAI crew.
    
    Captures key crew configuration that affects execution behavior:
    - process: Sequential or hierarchical execution flow
    - memory: Whether memory (short/long-term, entity) is enabled
    - cache: Whether tool result caching is enabled
    - planning: Whether agent planning is enabled
    - verbose: Logging verbosity level
    - max_rpm: Rate limiting configuration
    - agent_count/task_count: Crew composition
    
    See: https://docs.crewai.com/en/concepts/crews#crew-attributes
    """
    if not crew:
        return {}
    
    details: dict[str, Any] = {}
    
    # Crew identification
    crew_name = _safe_getattr(crew, "name", None)
    if crew_name:
        details["name"] = crew_name
    
    crew_id = _safe_getattr(crew, "id", None)
    if crew_id:
        details["id"] = str(crew_id)
    
    # Process type (sequential vs hierarchical) - critical for understanding execution
    process = _safe_getattr(crew, "process", None)
    if process is not None:
        # Process is an enum, get its value
        if hasattr(process, "value"):
            details["process"] = process.value
        elif hasattr(process, "name"):
            details["process"] = process.name
        else:
            details["process"] = str(process)
    
    # Memory configuration
    memory = _safe_getattr(crew, "memory", None)
    if memory is not None:
        details["memory_enabled"] = bool(memory)
    
    # Cache configuration
    cache = _safe_getattr(crew, "cache", None)
    if cache is not None:
        details["cache_enabled"] = bool(cache)
    
    # Planning configuration
    planning = _safe_getattr(crew, "planning", None)
    if planning is not None:
        details["planning_enabled"] = bool(planning)
    
    # Verbose/logging level
    verbose = _safe_getattr(crew, "verbose", None)
    if verbose is not None:
        details["verbose"] = bool(verbose)
    
    # Rate limiting
    max_rpm = _safe_getattr(crew, "max_rpm", None)
    if max_rpm is not None:
        details["max_rpm"] = max_rpm
    
    # Crew composition
    agents = _safe_getattr(crew, "agents", None)
    if agents and isinstance(agents, (list, tuple)):
        details["agent_count"] = len(agents)
        # Extract agent roles for quick reference
        agent_roles = []
        for agent in agents:
            role = _safe_getattr(agent, "role", None)
            if role:
                agent_roles.append(role)
        if agent_roles:
            details["agent_roles"] = agent_roles
    
    tasks = _safe_getattr(crew, "tasks", None)
    if tasks and isinstance(tasks, (list, tuple)):
        details["task_count"] = len(tasks)
    
    # Manager agent (for hierarchical process)
    manager_agent = _safe_getattr(crew, "manager_agent", None)
    if manager_agent:
        manager_role = _safe_getattr(manager_agent, "role", None)
        if manager_role:
            details["manager_agent_role"] = manager_role
    
    # Knowledge sources
    knowledge_sources = _safe_getattr(crew, "knowledge_sources", None)
    if knowledge_sources and isinstance(knowledge_sources, (list, tuple)):
        details["knowledge_source_count"] = len(knowledge_sources)
    
    return details


def _extract_task_details(task: Any) -> dict[str, Any]:
    """Extract detailed task attributes from CrewAI task.
    
    Captures key task configuration that affects execution behavior:
    - context: Other tasks whose outputs are used (drives implicit handoffs)
    - tools: Task-specific tools available
    - human_input: Whether human review is required
    - async_execution: Whether task runs asynchronously
    - expected_output: What the task should produce
    - guardrails: Validation functions applied to output
    - output_file: File path for task output
    
    See: https://docs.crewai.com/en/concepts/tasks#task-attributes
    """
    if not task:
        return {}
    
    details: dict[str, Any] = {}
    
    # Core identification
    task_id = _safe_getattr(task, "id", None) or _safe_getattr(task, "name", None)
    if task_id:
        details["id"] = task_id
    
    description = _safe_getattr(task, "description", None)
    if description:
        details["description"] = description
    
    expected_output = _safe_getattr(task, "expected_output", None)
    if expected_output:
        details["expected_output"] = expected_output
    
    # Execution configuration
    async_execution = _safe_getattr(task, "async_execution", None)
    if async_execution is not None:
        details["async_execution"] = bool(async_execution)
    
    human_input = _safe_getattr(task, "human_input", None)
    if human_input is not None:
        details["human_input"] = bool(human_input)
    
    # Context tasks (critical for implicit handoff tracking)
    context = _safe_getattr(task, "context", None)
    if context:
        context_task_ids = []
        if isinstance(context, (list, tuple)):
            for ctx_task in context:
                ctx_id = _safe_getattr(ctx_task, "id", None) or _safe_getattr(ctx_task, "name", None)
                if ctx_id:
                    context_task_ids.append(ctx_id)
        if context_task_ids:
            details["context_task_ids"] = context_task_ids
    
    # Task-specific tools
    tools = _safe_getattr(task, "tools", None)
    if tools:
        tool_names = []
        for tool in tools:
            if hasattr(tool, "name"):
                tool_names.append(tool.name)
            elif hasattr(tool, "__name__"):
                tool_names.append(tool.__name__)
            elif isinstance(tool, str):
                tool_names.append(tool)
        if tool_names:
            details["tools"] = tool_names
    
    # Output configuration
    output_file = _safe_getattr(task, "output_file", None)
    if output_file:
        details["output_file"] = output_file
    
    # Guardrails (validation)
    guardrails = _safe_getattr(task, "guardrails", None) or _safe_getattr(task, "guardrail", None)
    if guardrails:
        details["has_guardrails"] = True
        guardrail_max_retries = _safe_getattr(task, "guardrail_max_retries", None)
        if guardrail_max_retries is not None:
            details["guardrail_max_retries"] = guardrail_max_retries
    
    return details


def _extract_token_usage(response: Any) -> Optional[dict[str, int]]:
    """Extract token usage from LLM response.
    
    Supports multiple formats:
    - OpenAI/LiteLLM: response.usage.{prompt_tokens, completion_tokens, total_tokens}
    - usage_metadata: response.usage_metadata.{input_tokens, output_tokens}
    - Direct attributes: response.{prompt_tokens, completion_tokens, total_tokens}
    
    Returns:
        Dict with token counts or None if not available
    """
    if not response:
        return None
    
    result: dict[str, int] = {}
    
    # Try response.usage (OpenAI/LiteLLM format)
    usage = _safe_getattr(response, "usage", None)
    if usage:
        prompt = _safe_getattr(usage, "prompt_tokens", None)
        completion = _safe_getattr(usage, "completion_tokens", None)
        total = _safe_getattr(usage, "total_tokens", None)
        
        # Also check for input_tokens/output_tokens naming
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
    
    # Try response.usage_metadata (some providers)
    if not result:
        usage_meta = _safe_getattr(response, "usage_metadata", None)
        if usage_meta:
            input_tokens = _safe_getattr(usage_meta, "input_tokens", None)
            output_tokens = _safe_getattr(usage_meta, "output_tokens", None)
            
            if input_tokens is not None:
                result["prompt_tokens"] = int(input_tokens)
            if output_tokens is not None:
                result["completion_tokens"] = int(output_tokens)
            if input_tokens is not None and output_tokens is not None:
                result["total_tokens"] = int(input_tokens) + int(output_tokens)
    
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
    
    return result if result else None


def _base(
    run: "ArzuleRun",
    *,
    span_id: Optional[str],
    parent_span_id: Optional[str],
    async_id: Optional[str] = None,
    causal_parents: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build base event fields with optional async support."""
    event = {
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
    
    # Add async fields if provided
    if async_id:
        event["async_id"] = async_id
    if causal_parents:
        event["causal_parents"] = causal_parents
    
    return event


# =============================================================================
# Event Listener Normalization (Crew/Agent/Task lifecycle events)
# =============================================================================


def evt_from_crewai_event(
    run: "ArzuleRun",
    event: Any,
    *,
    parent_span_id: Optional[str] = None,
    task_key: Optional[str] = None,
) -> dict[str, Any]:
    """
    Convert a CrewAI event bus event to a TraceEvent.

    Args:
        run: The active ArzuleRun
        event: CrewAI event object
        parent_span_id: Optional explicit parent span (for concurrent task tracking)
        task_key: Optional task key for correlation in concurrent mode

    Returns:
        TraceEvent dict
    """
    event_class = event.__class__.__name__

    # Map event class names to our event types
    event_type_map = {
        # Crew lifecycle
        "CrewKickoffStartedEvent": "crew.kickoff.start",
        "CrewKickoffCompletedEvent": "crew.kickoff.complete",
        "CrewKickoffFailedEvent": "crew.kickoff.failed",
        # Agent lifecycle
        "AgentExecutionStartedEvent": "agent.execution.start",
        "AgentExecutionCompletedEvent": "agent.execution.complete",
        "AgentExecutionFailedEvent": "agent.execution.failed",
        "AgentExecutionErrorEvent": "agent.execution.error",
        # Task lifecycle
        "TaskStartedEvent": "task.start",
        "TaskCompletedEvent": "task.complete",
        "TaskFailedEvent": "task.failed",
        # Tool usage (CrewAI 1.7.x)
        "ToolUsageStartedEvent": "tool.call.start",
        "ToolUsageFinishedEvent": "tool.call.end",
        "ToolUsageErrorEvent": "tool.call.error",
        "ToolUsageEvent": "tool.call",
        # LLM calls (CrewAI 1.7.x)
        "LLMCallStartedEvent": "llm.call.start",
        "LLMCallCompletedEvent": "llm.call.end",
        "LLMCallFailedEvent": "llm.call.error",
        # Flow events
        "FlowStartedEvent": "flow.start",
        "FlowFinishedEvent": "flow.complete",
        "FlowCreatedEvent": "flow.created",
        "FlowPausedEvent": "flow.paused",
        "MethodExecutionStartedEvent": "flow.method.start",
        "MethodExecutionFinishedEvent": "flow.method.complete",
        "MethodExecutionFailedEvent": "flow.method.failed",
        "MethodExecutionPausedEvent": "flow.method.paused",
        # Human feedback events (Flow HITL)
        "HumanFeedbackRequestedEvent": "flow.human_feedback.requested",
        "HumanFeedbackReceivedEvent": "flow.human_feedback.received",
        # Memory events
        "MemoryQueryStartedEvent": "memory.query.start",
        "MemoryQueryCompletedEvent": "memory.query.end",
        "MemoryQueryFailedEvent": "memory.query.error",
        "MemorySaveStartedEvent": "memory.save.start",
        "MemorySaveCompletedEvent": "memory.save.end",
        "MemorySaveFailedEvent": "memory.save.error",
        # Knowledge events
        "KnowledgeQueryStartedEvent": "knowledge.query.start",
        "KnowledgeQueryCompletedEvent": "knowledge.query.end",
        "KnowledgeQueryFailedEvent": "knowledge.query.error",
        "KnowledgeRetrievalStartedEvent": "knowledge.retrieval.start",
        "KnowledgeRetrievalCompletedEvent": "knowledge.retrieval.end",
        # Reasoning events
        "AgentReasoningStartedEvent": "agent.reasoning.start",
        "AgentReasoningCompletedEvent": "agent.reasoning.end",
        "AgentReasoningFailedEvent": "agent.reasoning.error",
        # A2A (Agent-to-Agent) delegation events
        "A2ADelegationStartedEvent": "a2a.delegation.start",
        "A2ADelegationCompletedEvent": "a2a.delegation.complete",
        "A2AConversationStartedEvent": "a2a.conversation.start",
        "A2AConversationCompletedEvent": "a2a.conversation.complete",
        "A2AMessageSentEvent": "a2a.message.sent",
        "A2AResponseReceivedEvent": "a2a.response.received",
    }

    event_type = event_type_map.get(event_class, f"crewai.{event_class}")

    # Determine status
    status = "ok"
    if "Failed" in event_class or "Error" in event_class:
        status = "error"

    # Extract agent/task/tool info
    agent = _safe_getattr(event, "agent", None)
    task = _safe_getattr(event, "task", None)
    crew = _safe_getattr(event, "crew", None)

    agent_info = _extract_agent_info(agent)
    
    # Fallback: check for agent_role/agent_id directly on event (e.g., LLM events)
    # CrewAI's LLM events set agent_role/agent_id directly from from_agent, not as nested object
    if not agent_info:
        agent_role = _safe_getattr(event, "agent_role", None)
        if agent_role:
            agent_id = _safe_getattr(event, "agent_id", None)
            agent_info = {
                "id": agent_id or f"crewai:role:{agent_role}",
                "role": agent_role,
            }
    task_id, task_desc = _extract_task_info(task)

    # Tool info for tool events
    # CrewAI uses "tool_args" for tool input, fall back to "tool_input" for compatibility
    tool_name = _safe_getattr(event, "tool_name", None)
    tool_input = _safe_getattr(event, "tool_args", None)
    if tool_input is None:
        tool_input = _safe_getattr(event, "tool_input", None)
    tool_output = _safe_getattr(event, "tool_output", None) or _safe_getattr(event, "output", None)

    # Build summary
    summary_parts = [event_type]
    if agent_info:
        summary_parts.append(f"agent={agent_info['role']}")
    if task_id:
        summary_parts.append(f"task={task_id}")
    if tool_name:
        summary_parts.append(f"tool={tool_name}")
    summary = " ".join(summary_parts)

    # Extract result/error for completed/failed events
    payload: dict[str, Any] = {}
    attrs: dict[str, Any] = {}
    
    # Add extended agent info to attrs for queryable storage
    # (agent.id and agent.role go into indexed columns, rest goes here)
    if agent_info:
        if agent_info.get("allow_delegation") is not None:
            attrs["agent_allow_delegation"] = agent_info["allow_delegation"]
        if agent_info.get("goal"):
            attrs["agent_goal"] = agent_info["goal"]
        if agent_info.get("tools"):
            attrs["agent_tools"] = agent_info["tools"]
        if agent_info.get("max_iter") is not None:
            attrs["agent_max_iter"] = agent_info["max_iter"]
        if agent_info.get("max_execution_time") is not None:
            attrs["agent_max_execution_time"] = agent_info["max_execution_time"]
        if agent_info.get("verbose") is not None:
            attrs["agent_verbose"] = agent_info["verbose"]

    result = _safe_getattr(event, "result", None)
    if result is not None:
        payload["result"] = truncate_string(str(result), 1000)

    output = _safe_getattr(event, "output", None)
    if output is not None:
        payload["output"] = truncate_string(str(output), 1000)

    error = _safe_getattr(event, "error", None)
    if error is not None:
        attrs["error"] = truncate_string(str(error), 200)

    # Crew info (detailed extraction)
    if crew:
        crew_details = _extract_crew_details(crew)
        payload["crew"] = sanitize(crew_details)
        
        # Add key crew attributes to attrs for queryable storage
        if crew_details.get("name"):
            attrs["crew_name"] = crew_details["name"]
        if crew_details.get("process"):
            attrs["crew_process"] = crew_details["process"]
        if crew_details.get("memory_enabled"):
            attrs["crew_memory_enabled"] = True
        if crew_details.get("planning_enabled"):
            attrs["crew_planning_enabled"] = True
        if crew_details.get("agent_count"):
            attrs["crew_agent_count"] = crew_details["agent_count"]
        if crew_details.get("task_count"):
            attrs["crew_task_count"] = crew_details["task_count"]
        if crew_details.get("manager_agent_role"):
            attrs["crew_manager_role"] = crew_details["manager_agent_role"]

    # Task info in payload (detailed extraction)
    if task is not None:
        task_details = _extract_task_details(task)
        payload["task"] = sanitize(task_details)
        
        # Add key task attributes to attrs for queryable storage
        if task_details.get("async_execution"):
            attrs["task_async_execution"] = True
        if task_details.get("human_input"):
            attrs["task_human_input"] = True
        if task_details.get("context_task_ids"):
            attrs["task_context_ids"] = task_details["context_task_ids"]
        if task_details.get("tools"):
            attrs["task_tools"] = task_details["tools"]
        if task_details.get("has_guardrails"):
            attrs["task_has_guardrails"] = True
        if task_details.get("output_file"):
            attrs["task_output_file"] = task_details["output_file"]

    # Flow event info (for multi-crew orchestration)
    flow_name = _safe_getattr(event, "flow_name", None)
    method_name = _safe_getattr(event, "method_name", None)
    flow_state = _safe_getattr(event, "state", None)
    flow_params = _safe_getattr(event, "params", None)
    flow_inputs = _safe_getattr(event, "inputs", None)
    flow_id = _safe_getattr(event, "flow_id", None)
    
    if flow_name:
        attrs["flow_name"] = flow_name
        # Update summary for flow events
        if "flow" in event_type or "method" in event_type:
            summary_parts = [event_type, f"flow={flow_name}"]
            if method_name:
                summary_parts.append(f"method={method_name}")
            summary = " ".join(summary_parts)
    
    if method_name:
        attrs["method_name"] = method_name
    
    if flow_id:
        attrs["flow_id"] = flow_id
    
    # Flow state (convert Pydantic models to dict)
    if flow_state is not None:
        state_dict = None
        if hasattr(flow_state, "model_dump"):
            state_dict = flow_state.model_dump()
        elif hasattr(flow_state, "dict"):
            state_dict = flow_state.dict()
        elif isinstance(flow_state, dict):
            state_dict = flow_state
        if state_dict:
            payload["flow_state"] = sanitize(state_dict)
    
    # Flow method params
    if flow_params is not None:
        payload["flow_params"] = sanitize(flow_params)
    
    # Flow inputs (for FlowStartedEvent)
    if flow_inputs is not None:
        payload["flow_inputs"] = sanitize(flow_inputs)
    
    # Human feedback events (HITL)
    feedback_message = _safe_getattr(event, "message", None)
    feedback_text = _safe_getattr(event, "feedback", None)
    feedback_outcome = _safe_getattr(event, "outcome", None)
    feedback_emit = _safe_getattr(event, "emit", None)
    
    if feedback_message:
        attrs["feedback_message"] = truncate_string(feedback_message, 200)
    if feedback_text:
        payload["feedback"] = truncate_string(feedback_text, 1000)
    if feedback_outcome:
        attrs["feedback_outcome"] = feedback_outcome
    if feedback_emit:
        attrs["feedback_emit_options"] = feedback_emit

    # Tool info in payload/attrs
    if tool_name:
        attrs["tool_name"] = tool_name
        
        # Mark delegation tools as handoffs for visual distinction in UI
        if is_delegation_tool(tool_name):
            attrs["is_handoff"] = True
        
        # Add detection fields for forensics (tool events)
        if tool_input is not None:
            input_hash = _compute_input_hash(tool_input)
            input_keys = _extract_input_keys(tool_input)
            if input_hash:
                attrs["tool_input_hash"] = input_hash
            if input_keys:
                attrs["tool_input_keys"] = input_keys
    
    if tool_input is not None:
        payload["tool_input"] = sanitize(tool_input)
    if tool_output is not None:
        payload["tool_output"] = sanitize(tool_output)

    # LLM event info in payload
    messages = _safe_getattr(event, "messages", None)
    if messages is not None:
        payload["messages"] = _truncate_messages(messages)
        attrs["message_count"] = len(messages) if isinstance(messages, list) else 0

    response = _safe_getattr(event, "response", None)
    if response is not None:
        content = _safe_getattr(response, "content", None)
        if content is None:
            content = str(response)
        payload["response"] = truncate_string(str(content), 2000)
        
        # Extract token usage from LLM response (LiteLLM/OpenAI format)
        token_usage = _extract_token_usage(response)
        if token_usage:
            attrs.update(token_usage)

    # LLM model info
    model = _safe_getattr(event, "model", None)
    if model:
        attrs["llm_model"] = model

    # Memory event info
    memory_query = _safe_getattr(event, "query", None)
    memory_limit = _safe_getattr(event, "limit", None)
    memory_score_threshold = _safe_getattr(event, "score_threshold", None)
    memory_results = _safe_getattr(event, "results", None)
    memory_query_time = _safe_getattr(event, "query_time_ms", None)
    memory_value = _safe_getattr(event, "value", None)
    memory_metadata = _safe_getattr(event, "metadata", None)
    memory_save_time = _safe_getattr(event, "save_time_ms", None)
    memory_content = _safe_getattr(event, "memory_content", None)
    memory_retrieval_time = _safe_getattr(event, "retrieval_time_ms", None)
    
    if memory_query and "memory" in event_type:
        attrs["memory_query"] = truncate_string(memory_query, 200)
    if memory_limit is not None:
        attrs["memory_limit"] = memory_limit
    if memory_score_threshold is not None:
        attrs["memory_score_threshold"] = memory_score_threshold
    if memory_results is not None:
        payload["memory_results"] = sanitize(memory_results)
    if memory_query_time is not None:
        attrs["memory_query_time_ms"] = memory_query_time
    if memory_value:
        payload["memory_value"] = truncate_string(memory_value, 500)
    if memory_metadata:
        payload["memory_metadata"] = sanitize(memory_metadata)
    if memory_save_time is not None:
        attrs["memory_save_time_ms"] = memory_save_time
    if memory_content:
        payload["memory_content"] = truncate_string(memory_content, 1000)
    if memory_retrieval_time is not None:
        attrs["memory_retrieval_time_ms"] = memory_retrieval_time

    # Knowledge event info
    knowledge_query = _safe_getattr(event, "query", None)
    knowledge_task_prompt = _safe_getattr(event, "task_prompt", None)
    knowledge_retrieved = _safe_getattr(event, "retrieved_knowledge", None)
    
    if knowledge_query and "knowledge" in event_type:
        attrs["knowledge_query"] = truncate_string(knowledge_query, 200)
    if knowledge_task_prompt:
        payload["knowledge_task_prompt"] = truncate_string(knowledge_task_prompt, 500)
    if knowledge_retrieved:
        payload["knowledge_retrieved"] = truncate_string(knowledge_retrieved, 1000)

    # A2A (Agent-to-Agent) delegation event info
    if "a2a" in event_type:
        # Delegation info
        a2a_endpoint = _safe_getattr(event, "endpoint", None)
        a2a_task_description = _safe_getattr(event, "task_description", None)
        a2a_agent_id = _safe_getattr(event, "agent_id", None)
        a2a_is_multiturn = _safe_getattr(event, "is_multiturn", None)
        a2a_turn_number = _safe_getattr(event, "turn_number", None)
        a2a_status = _safe_getattr(event, "status", None)
        a2a_result = _safe_getattr(event, "result", None)
        a2a_final_result = _safe_getattr(event, "final_result", None)
        a2a_error = _safe_getattr(event, "error", None)
        a2a_message = _safe_getattr(event, "message", None)
        a2a_response = _safe_getattr(event, "response", None)
        a2a_agent_name = _safe_getattr(event, "a2a_agent_name", None)
        a2a_total_turns = _safe_getattr(event, "total_turns", None)
        a2a_agent_role = _safe_getattr(event, "agent_role", None)
        
        if a2a_endpoint:
            attrs["a2a_endpoint"] = a2a_endpoint
        if a2a_agent_id:
            attrs["a2a_agent_id"] = a2a_agent_id
        if a2a_agent_name:
            attrs["a2a_agent_name"] = a2a_agent_name
        if a2a_is_multiturn is not None:
            attrs["a2a_is_multiturn"] = a2a_is_multiturn
        if a2a_turn_number is not None:
            attrs["a2a_turn_number"] = a2a_turn_number
        if a2a_total_turns is not None:
            attrs["a2a_total_turns"] = a2a_total_turns
        if a2a_status:
            attrs["a2a_status"] = a2a_status
        if a2a_agent_role:
            attrs["a2a_agent_role"] = a2a_agent_role
        
        if a2a_task_description:
            payload["a2a_task_description"] = truncate_string(a2a_task_description, 500)
        if a2a_result:
            payload["a2a_result"] = truncate_string(str(a2a_result), 1000)
        if a2a_final_result:
            payload["a2a_final_result"] = truncate_string(str(a2a_final_result), 1000)
        if a2a_error:
            attrs["a2a_error"] = truncate_string(str(a2a_error), 200)
        if a2a_message:
            payload["a2a_message"] = truncate_string(a2a_message, 1000)
        if a2a_response:
            payload["a2a_response"] = truncate_string(a2a_response, 1000)
        
        # Update summary for A2A events
        summary_parts = [event_type]
        if a2a_agent_id:
            summary_parts.append(f"agent={a2a_agent_id}")
        if a2a_turn_number is not None:
            summary_parts.append(f"turn={a2a_turn_number}")
        if a2a_status:
            summary_parts.append(f"status={a2a_status}")
        summary = " ".join(summary_parts)

    # Determine parent span:
    # 1. Use explicit parent_span_id if provided (concurrent mode)
    # 2. Try task-based lookup if task_key provided
    # 3. Fall back to legacy sequential mode
    effective_parent_span = parent_span_id
    if effective_parent_span is None and task_key:
        effective_parent_span = run.get_task_parent_span(task_key=task_key)
    if effective_parent_span is None:
        # Extract agent key for agent-based task lookup
        agent_key = f"agent:{agent_info['role']}" if agent_info and agent_info.get('role') else None
        if agent_key:
            effective_parent_span = run.get_task_parent_span(agent_key=agent_key)
    if effective_parent_span is None:
        effective_parent_span = run.current_parent_span_id()

    return {
        "schema_version": "trace_event.v0_1",
        "run_id": run.run_id,
        "tenant_id": run.tenant_id,
        "project_id": run.project_id,
        "trace_id": run.trace_id,
        "span_id": new_span_id(),
        "parent_span_id": effective_parent_span,
        "seq": run.next_seq(),
        "ts": run.now(),
        "workstream_id": None,
        "agent": agent_info,
        "task_id": task_id,
        "event_type": event_type,
        "status": status,
        "summary": summary,
        "attrs_compact": attrs,
        "payload": payload,
        "raw_ref": {"storage": "inline"},
    }


# =============================================================================
# Tool Hook Normalization
# =============================================================================


def evt_tool_start(run: "ArzuleRun", context: Any, span_id: str) -> dict[str, Any]:
    """
    Create event for tool call start.

    Args:
        run: The active ArzuleRun
        context: CrewAI tool call context
        span_id: The span ID for this tool call

    Returns:
        TraceEvent dict
    """
    tool_name = _safe_getattr(context, "tool_name", "unknown_tool")
    tool_input = _safe_getattr(context, "tool_input", {})
    agent = _safe_getattr(context, "agent", None)

    agent_info = _extract_agent_info(agent)

    # Extract handoff_key if present
    handoff_key = None
    if isinstance(tool_input, dict):
        arz = tool_input.get("arzule", {})
        handoff_key = arz.get("handoff_key") if isinstance(arz, dict) else None

    # Compute detection fields for forensics
    input_hash = _compute_input_hash(tool_input)
    input_keys = _extract_input_keys(tool_input)

    attrs = {
        "tool_name": tool_name,
        "handoff_key": handoff_key,
    }
    
    # Add detection fields if available
    if input_hash:
        attrs["tool_input_hash"] = input_hash
    if input_keys:
        attrs["tool_input_keys"] = input_keys

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": agent_info,
        "event_type": "tool.call.start",
        "status": "ok",
        "summary": f"tool call: {tool_name}",
        "attrs_compact": attrs,
        "payload": {
            "tool_input": sanitize(tool_input),
        },
    }


def evt_tool_end(run: "ArzuleRun", context: Any, span_id: Optional[str]) -> dict[str, Any]:
    """
    Create event for tool call end.

    Args:
        run: The active ArzuleRun
        context: CrewAI tool call context
        span_id: The span ID for this tool call

    Returns:
        TraceEvent dict
    """
    tool_name = _safe_getattr(context, "tool_name", "unknown_tool")
    # Try both attribute names for tool output
    tool_result = _safe_getattr(context, "tool_result", None)
    if tool_result is None:
        tool_result = _safe_getattr(context, "tool_output", None)
    tool_error = _safe_getattr(context, "error", None) or _safe_getattr(context, "exception", None)
    agent = _safe_getattr(context, "agent", None)
    tool_input = _safe_getattr(context, "tool_input", {})

    agent_info = _extract_agent_info(agent)
    status = "error" if tool_error else "ok"

    # Extract handoff_key if present
    handoff_key = None
    if isinstance(tool_input, dict):
        arz = tool_input.get("arzule", {})
        handoff_key = arz.get("handoff_key") if isinstance(arz, dict) else None

    # Compute detection fields for forensics
    input_hash = _compute_input_hash(tool_input)
    input_keys = _extract_input_keys(tool_input)

    attrs = {
        "tool_name": tool_name,
        "handoff_key": handoff_key,
    }
    
    # Add detection fields if available
    if input_hash:
        attrs["tool_input_hash"] = input_hash
    if input_keys:
        attrs["tool_input_keys"] = input_keys

    payload: dict[str, Any] = {
        "tool_input": sanitize(tool_input),
    }
    if tool_result is not None:
        payload["tool_result"] = sanitize(tool_result)
    if tool_error:
        payload["error"] = truncate_string(str(tool_error), 500)

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": agent_info,
        "event_type": "tool.call.end",
        "status": status,
        "summary": f"tool result: {tool_name}",
        "attrs_compact": attrs,
        "payload": payload,
    }


# =============================================================================
# LLM Hook Normalization
# =============================================================================


def evt_llm_start(run: "ArzuleRun", context: Any, span_id: str) -> dict[str, Any]:
    """
    Create event for LLM call start.

    Args:
        run: The active ArzuleRun
        context: CrewAI LLM call context
        span_id: The span ID for this LLM call

    Returns:
        TraceEvent dict
    """
    messages = _safe_getattr(context, "messages", [])
    agent = _safe_getattr(context, "agent", None)
    task = _safe_getattr(context, "task", None)

    agent_info = _extract_agent_info(agent)
    task_id, _ = _extract_task_info(task)

    # Summarize messages
    msg_count = len(messages) if isinstance(messages, list) else 0

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": agent_info,
        "task_id": task_id,
        "event_type": "llm.call.start",
        "status": "ok",
        "summary": f"llm call with {msg_count} messages",
        "attrs_compact": {
            "message_count": msg_count,
        },
        "payload": {
            "messages": _truncate_messages(messages),
        },
    }


def evt_llm_end(run: "ArzuleRun", context: Any, span_id: Optional[str]) -> dict[str, Any]:
    """
    Create event for LLM call end.

    Args:
        run: The active ArzuleRun
        context: CrewAI LLM call context
        span_id: The span ID for this LLM call

    Returns:
        TraceEvent dict
    """
    response = _safe_getattr(context, "response", None)
    agent = _safe_getattr(context, "agent", None)
    task = _safe_getattr(context, "task", None)
    error = _safe_getattr(context, "error", None) or _safe_getattr(context, "exception", None)

    agent_info = _extract_agent_info(agent)
    task_id, _ = _extract_task_info(task)
    status = "error" if error else "ok"

    payload: dict[str, Any] = {}
    attrs: dict[str, Any] = {}
    
    if response is not None:
        # Extract content from response object or use as string
        content = _safe_getattr(response, "content", None)
        if content is None:
            content = str(response)
        payload["response"] = truncate_string(str(content), 2000)
        
        # Extract token usage for per-agent tracking
        token_usage = _extract_token_usage(response)
        if token_usage:
            attrs.update(token_usage)

    if error:
        payload["error"] = truncate_string(str(error), 500)

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": agent_info,
        "task_id": task_id,
        "event_type": "llm.call.end",
        "status": status,
        "summary": "llm response received",
        "attrs_compact": attrs,
        "payload": payload,
    }


def _truncate_messages(messages: Any, max_messages: int = 10) -> list[dict[str, Any]]:
    """Truncate message list for payload."""
    if not isinstance(messages, list):
        return []

    result = []
    for i, msg in enumerate(messages[:max_messages]):
        if isinstance(msg, dict):
            result.append({
                "role": msg.get("role", "unknown"),
                "content": truncate_string(str(msg.get("content", "")), 500),
            })
        else:
            # Try to extract from object
            result.append({
                "role": _safe_getattr(msg, "role", "unknown"),
                "content": truncate_string(str(_safe_getattr(msg, "content", msg)), 500),
            })

    if len(messages) > max_messages:
        result.append({"_truncated": f"{len(messages) - max_messages} more messages"})

    return result


# =============================================================================
# Async Boundary Events (for async_execution=True tasks)
# =============================================================================


def evt_async_spawn(
    run: "ArzuleRun",
    task_key: str,
    async_id: str,
    parent_span_id: Optional[str],
    task_description: Optional[str] = None,
    agent_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Create async.spawn event when parent spawns an async child task.

    The causal_parents links this spawn to the spawning context.

    Args:
        run: The active ArzuleRun
        task_key: Unique identifier for the async task
        async_id: The async correlation ID
        parent_span_id: The parent span that spawned this async task
        task_description: Optional task description
        agent_info: Optional agent info dict

    Returns:
        TraceEvent dict
    """
    span_id = new_span_id()
    causal_parents = [parent_span_id] if parent_span_id else []

    return {
        **_base(
            run,
            span_id=span_id,
            parent_span_id=parent_span_id,
            async_id=async_id,
            causal_parents=causal_parents,
        ),
        "agent": agent_info,
        "event_type": "async.spawn",
        "status": "ok",
        "summary": f"async task spawned: {task_key}",
        "attrs_compact": {
            "async_id": async_id,
            "task_key": task_key,
            "causal_parents": causal_parents,
        },
        "payload": {
            "task_description": truncate_string(task_description, 500) if task_description else None,
        },
    }


def evt_async_join(
    run: "ArzuleRun",
    task_key: str,
    async_id: str,
    parent_span_id: Optional[str],
    status: str = "ok",
    result_summary: Optional[str] = None,
    agent_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Create async.join event when an async context completes.

    Args:
        run: The active ArzuleRun
        task_key: The task identifier
        async_id: The async correlation ID
        parent_span_id: The parent span
        status: Completion status (ok or error)
        result_summary: Optional summary of the result
        agent_info: Optional agent info dict

    Returns:
        TraceEvent dict
    """
    span_id = new_span_id()

    return {
        **_base(
            run,
            span_id=span_id,
            parent_span_id=parent_span_id,
            async_id=async_id,
        ),
        "agent": agent_info,
        "event_type": "async.join",
        "status": status,
        "summary": result_summary or f"async task completed: {task_key}",
        "attrs_compact": {
            "async_id": async_id,
            "task_key": task_key,
        },
        "payload": {},
    }


# =============================================================================
# Flow Events (for multi-crew orchestration)
# =============================================================================


def evt_flow_start(
    run: "ArzuleRun",
    flow_name: str,
    span_id: str,
    parent_span_id: Optional[str],
    inputs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Create event for flow start.

    Args:
        run: The active ArzuleRun
        flow_name: Name of the flow
        span_id: The span ID for this flow
        parent_span_id: The parent span (usually root span)
        inputs: Optional flow inputs

    Returns:
        TraceEvent dict
    """
    attrs = {
        "flow_name": flow_name,
    }
    
    payload: dict[str, Any] = {}
    if inputs:
        payload["inputs"] = sanitize(inputs)

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "flow.start",
        "status": "ok",
        "summary": f"flow started: {flow_name}",
        "attrs_compact": attrs,
        "payload": payload,
    }


def evt_flow_end(
    run: "ArzuleRun",
    flow_name: str,
    span_id: str,
    parent_span_id: Optional[str],
    status: str = "ok",
    result: Optional[Any] = None,
    state: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Create event for flow completion.

    Args:
        run: The active ArzuleRun
        flow_name: Name of the flow
        span_id: The span ID for this event
        parent_span_id: The parent span (flow span)
        status: Completion status (ok or error)
        result: Optional flow result
        state: Optional final flow state

    Returns:
        TraceEvent dict
    """
    attrs = {
        "flow_name": flow_name,
    }
    
    payload: dict[str, Any] = {}
    if result is not None:
        payload["result"] = truncate_string(str(result), 1000)
    if state:
        payload["final_state"] = sanitize(state)

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "flow.complete",
        "status": status,
        "summary": f"flow completed: {flow_name}",
        "attrs_compact": attrs,
        "payload": payload,
    }


def evt_method_start(
    run: "ArzuleRun",
    flow_name: str,
    method_name: str,
    span_id: str,
    parent_span_id: Optional[str],
    params: Optional[dict[str, Any]] = None,
    state: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Create event for flow method execution start.

    Args:
        run: The active ArzuleRun
        flow_name: Name of the flow
        method_name: Name of the method being executed
        span_id: The span ID for this method
        parent_span_id: The parent span (flow span)
        params: Optional method parameters
        state: Optional current flow state

    Returns:
        TraceEvent dict
    """
    attrs = {
        "flow_name": flow_name,
        "method_name": method_name,
    }
    
    payload: dict[str, Any] = {}
    if params:
        payload["params"] = sanitize(params)
    if state:
        payload["state"] = sanitize(state)

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "flow.method.start",
        "status": "ok",
        "summary": f"method started: {flow_name}.{method_name}",
        "attrs_compact": attrs,
        "payload": payload,
    }


def evt_method_end(
    run: "ArzuleRun",
    flow_name: str,
    method_name: str,
    span_id: str,
    parent_span_id: Optional[str],
    status: str = "ok",
    result: Optional[Any] = None,
    state: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create event for flow method execution completion.

    Args:
        run: The active ArzuleRun
        flow_name: Name of the flow
        method_name: Name of the method that completed
        span_id: The span ID for this event
        parent_span_id: The parent span (method start span)
        status: Completion status (ok or error)
        result: Optional method result
        state: Optional updated flow state
        error: Optional error message if failed

    Returns:
        TraceEvent dict
    """
    attrs = {
        "flow_name": flow_name,
        "method_name": method_name,
    }
    
    payload: dict[str, Any] = {}
    if result is not None:
        payload["result"] = truncate_string(str(result), 1000)
    if state:
        payload["state"] = sanitize(state)
    if error:
        attrs["error"] = truncate_string(error, 200)

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "flow.method.complete" if status == "ok" else "flow.method.failed",
        "status": status,
        "summary": f"method {'completed' if status == 'ok' else 'failed'}: {flow_name}.{method_name}",
        "attrs_compact": attrs,
        "payload": payload,
    }

