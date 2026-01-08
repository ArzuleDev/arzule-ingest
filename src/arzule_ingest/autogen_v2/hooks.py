"""Instrumentation hooks for AutoGen v0.7+ (autogen-core and autogen-agentchat).

This module patches the new AutoGen API to capture:
- Agent lifecycle events (on_messages, on_reset, etc.)
- Model calls (via ChatCompletionClient)
- Tool executions
- Agent events and messages
- Telemetry traces

Thread Safety:
- Uses contextvars from autogen-core where possible
- Falls back to cached run_id when needed
"""

from __future__ import annotations

import functools
import sys
import threading
from typing import Any, Callable, Optional, TYPE_CHECKING

from ..ids import new_span_id
from ..logger import log_event_dropped
from ..run import current_run
from .normalize import (
    evt_agent_start,
    evt_agent_end,
    evt_message_event,
    evt_chat_message,
    evt_model_call_start,
    evt_model_call_end,
    evt_tool_call_start,
    evt_tool_call_end,
)
from .spanctx import (
    start_agent_span,
    end_agent_span,
    get_current_span_id,
)

if TYPE_CHECKING:
    from ..run import ArzuleRun

# Store original methods for restoration
_original_methods: dict[str, Any] = {}

# =============================================================================
# Cached run_id for thread fallback (module-level)
# =============================================================================

_cached_run_id: Optional[str] = None
_cached_run_id_lock = threading.Lock()


def _cache_run_id(run_id: str) -> None:
    """Cache the run_id for thread-safe fallback lookup."""
    global _cached_run_id
    with _cached_run_id_lock:
        _cached_run_id = run_id


def _clear_cached_run_id() -> None:
    """Clear the cached run_id."""
    global _cached_run_id
    with _cached_run_id_lock:
        _cached_run_id = None


def _get_cached_run_id() -> Optional[str]:
    """Get the cached run_id (thread-safe)."""
    with _cached_run_id_lock:
        return _cached_run_id


def _get_run_with_fallback(hook_name: str) -> Optional["ArzuleRun"]:
    """Get run from ContextVar, falling back to cached run_id.
    
    Args:
        hook_name: Name of the hook (for logging if dropped)
        
    Returns:
        The ArzuleRun instance, or None if not recoverable
    """
    # Try ContextVar first
    run = current_run()
    if run:
        # Update cache when we have a valid run
        _cache_run_id(run.run_id)
        return run
    
    # Fallback: try global registry with cached run_id
    cached_id = _get_cached_run_id()
    if cached_id:
        run = current_run(run_id_hint=cached_id)
        if run:
            return run
    
    # Log the drop (only if we had a cached_id, meaning we were expecting a run)
    if cached_id:
        log_event_dropped(
            reason="no_active_run_and_fallback_failed",
            event_class=f"autogen_v2.{hook_name}",
            extra={"cached_run_id": cached_id}
        )
    return None


def _get_agent_name(agent: Any) -> str:
    """Safely get agent name."""
    try:
        return getattr(agent, "name", None) or "unknown"
    except Exception:
        return "unknown"


# =============================================================================
# Agent Lifecycle Hooks
# =============================================================================


def _patched_on_messages(
    original_on_messages: Callable,
) -> Callable:
    """Create patched on_messages method for agent message handling."""
    
    @functools.wraps(original_on_messages)
    async def wrapper(self, messages, cancellation_token):
        run = _get_run_with_fallback("on_messages")
        
        if not run:
            return await original_on_messages(self, messages, cancellation_token)
        
        agent_name = _get_agent_name(self)
        span_id = start_agent_span(run, agent_name)
        span_pushed = False
        
        try:
            # Emit agent start event
            start_evt = evt_agent_start(
                run=run,
                agent=self,
                messages=messages,
                span_id=span_id,
            )
            run.emit(start_evt)
            run.push_span(span_id)
            span_pushed = True
            
            # Emit events for incoming messages
            for msg in messages:
                try:
                    msg_evt = evt_chat_message(
                        run=run,
                        agent=self,
                        message=msg,
                        direction="incoming",
                        span_id=get_current_span_id(run),
                    )
                    run.emit(msg_evt)
                except Exception as e:
                    print(f"[arzule] Error emitting message event: {e}", file=sys.stderr)
            
        except Exception as e:
            print(f"[arzule] Error in agent start hook: {e}", file=sys.stderr)
        
        error = None
        result = None
        
        try:
            result = await original_on_messages(self, messages, cancellation_token)
            return result
        except Exception as e:
            error = e
            raise
        finally:
            if run:
                try:
                    # Emit event for response message
                    if result and hasattr(result, 'chat_message'):
                        msg_evt = evt_chat_message(
                            run=run,
                            agent=self,
                            message=result.chat_message,
                            direction="outgoing",
                            span_id=get_current_span_id(run),
                        )
                        run.emit(msg_evt)
                    
                    # Emit agent end event
                    end_evt = evt_agent_end(
                        run=run,
                        agent=self,
                        response=result,
                        span_id=span_id,
                        error=error,
                    )
                    run.emit(end_evt)
                    
                    # Only pop if we successfully pushed
                    if span_pushed:
                        run.pop_span()
                        end_agent_span(run, agent_name)
                    
                except Exception as e:
                    print(f"[arzule] Error in agent end hook: {e}", file=sys.stderr)
    
    return wrapper


def _patched_on_messages_stream(
    original_on_messages_stream: Callable,
) -> Callable:
    """Create patched on_messages_stream method for streaming agent responses."""
    
    @functools.wraps(original_on_messages_stream)
    async def wrapper(self, messages, cancellation_token):
        run = _get_run_with_fallback("on_messages_stream")
        
        if not run:
            async for item in original_on_messages_stream(self, messages, cancellation_token):
                yield item
            return
        
        agent_name = _get_agent_name(self)
        span_id = start_agent_span(run, agent_name)
        span_pushed = False
        
        try:
            # Emit agent start event
            start_evt = evt_agent_start(
                run=run,
                agent=self,
                messages=messages,
                span_id=span_id,
            )
            run.emit(start_evt)
            run.push_span(span_id)
            span_pushed = True
            
            # Emit events for incoming messages
            for msg in messages:
                try:
                    msg_evt = evt_chat_message(
                        run=run,
                        agent=self,
                        message=msg,
                        direction="incoming",
                        span_id=get_current_span_id(run),
                    )
                    run.emit(msg_evt)
                except Exception as e:
                    print(f"[arzule] Error emitting message event: {e}", file=sys.stderr)
            
        except Exception as e:
            print(f"[arzule] Error in agent stream start hook: {e}", file=sys.stderr)
        
        error = None
        final_response = None
        
        try:
            async for item in original_on_messages_stream(self, messages, cancellation_token):
                # Emit events for streamed items
                try:
                    # Check if it's a Response object (final item)
                    if hasattr(item, 'chat_message'):
                        final_response = item
                        msg_evt = evt_chat_message(
                            run=run,
                            agent=self,
                            message=item.chat_message,
                            direction="outgoing",
                            span_id=get_current_span_id(run),
                        )
                        run.emit(msg_evt)
                    else:
                        # It's an event or intermediate message
                        evt = evt_message_event(
                            run=run,
                            agent=self,
                            event=item,
                            span_id=get_current_span_id(run),
                        )
                        run.emit(evt)
                except Exception as e:
                    print(f"[arzule] Error emitting stream event: {e}", file=sys.stderr)
                
                yield item
                
        except Exception as e:
            error = e
            raise
        finally:
            if run:
                try:
                    # Emit agent end event
                    end_evt = evt_agent_end(
                        run=run,
                        agent=self,
                        response=final_response,
                        span_id=span_id,
                        error=error,
                    )
                    run.emit(end_evt)
                    
                    # Only pop if we successfully pushed
                    if span_pushed:
                        run.pop_span()
                        end_agent_span(run, agent_name)
                    
                except Exception as e:
                    print(f"[arzule] Error in agent stream end hook: {e}", file=sys.stderr)
    
    return wrapper


# =============================================================================
# Model Client Hooks
# =============================================================================


def _patched_create(
    original_create: Callable,
) -> Callable:
    """Create patched create method for ChatCompletionClient."""
    
    @functools.wraps(original_create)
    async def wrapper(self, messages, *args, **kwargs):
        run = _get_run_with_fallback("model_create")
        
        if not run:
            return await original_create(self, messages, *args, **kwargs)
        
        span_id = new_span_id()
        span_pushed = False
        
        try:
            # Emit model call start event
            start_evt = evt_model_call_start(
                run=run,
                model_client=self,
                messages=messages,
                span_id=span_id,
            )
            run.emit(start_evt)
            run.push_span(span_id)
            span_pushed = True
            
        except Exception as e:
            print(f"[arzule] Error in model start hook: {e}", file=sys.stderr)
        
        error = None
        result = None
        
        try:
            result = await original_create(self, messages, *args, **kwargs)
            return result
        except Exception as e:
            error = e
            raise
        finally:
            if run:
                try:
                    end_evt = evt_model_call_end(
                        run=run,
                        model_client=self,
                        response=result,
                        span_id=span_id,
                        error=error,
                    )
                    run.emit(end_evt)
                    
                    # Only pop if we successfully pushed
                    if span_pushed:
                        run.pop_span()
                    
                except Exception as e:
                    print(f"[arzule] Error in model end hook: {e}", file=sys.stderr)
    
    return wrapper


# =============================================================================
# Tool Execution Hooks
# =============================================================================


def _patched_run_tool(
    original_run_tool: Callable,
) -> Callable:
    """Create patched run method for tools."""
    
    @functools.wraps(original_run_tool)
    async def wrapper(self, args, cancellation_token):
        run = _get_run_with_fallback("tool_run")
        
        if not run:
            return await original_run_tool(self, args, cancellation_token)
        
        tool_name = getattr(self, 'name', 'unknown')
        span_id = new_span_id()
        span_pushed = False
        
        try:
            # Emit tool start event
            start_evt = evt_tool_call_start(
                run=run,
                tool=self,
                tool_name=tool_name,
                tool_input=args,
                span_id=span_id,
            )
            run.emit(start_evt)
            run.push_span(span_id)
            span_pushed = True
            
        except Exception as e:
            print(f"[arzule] Error in tool start hook: {e}", file=sys.stderr)
        
        error = None
        result = None
        
        try:
            result = await original_run_tool(self, args, cancellation_token)
            return result
        except Exception as e:
            error = e
            raise
        finally:
            if run:
                try:
                    end_evt = evt_tool_call_end(
                        run=run,
                        tool=self,
                        tool_name=tool_name,
                        tool_output=result,
                        span_id=span_id,
                        error=error,
                    )
                    run.emit(end_evt)
                    
                    # Only pop if we successfully pushed
                    if span_pushed:
                        run.pop_span()
                    
                except Exception as e:
                    print(f"[arzule] Error in tool end hook: {e}", file=sys.stderr)
    
    return wrapper


# =============================================================================
# Installation
# =============================================================================


def install_hooks(
    enable_message_hooks: bool = True,
    enable_llm_hooks: bool = True,
    enable_tool_hooks: bool = True,
    enable_agent_hooks: bool = True,
    enable_telemetry: bool = True,
) -> None:
    """
    Install monkey-patching hooks for AutoGen v0.7+.
    
    Args:
        enable_message_hooks: Install message handling hooks
        enable_llm_hooks: Install model client hooks
        enable_tool_hooks: Install tool execution hooks
        enable_agent_hooks: Install agent lifecycle hooks
        enable_telemetry: Hook into autogen-core telemetry (if available)
    """
    global _original_methods
    
    try:
        from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
    except ImportError:
        print("[arzule] autogen-agentchat not installed, skipping hooks", file=sys.stderr)
        return
    
    installed = []
    
    # Patch agent lifecycle methods
    if enable_agent_hooks:
        # Patch BaseChatAgent.on_messages
        if hasattr(BaseChatAgent, "on_messages"):
            if not hasattr(BaseChatAgent.on_messages, "_arzule_patched"):
                _original_methods["on_messages"] = BaseChatAgent.on_messages
                BaseChatAgent.on_messages = _patched_on_messages(BaseChatAgent.on_messages)
                BaseChatAgent.on_messages._arzule_patched = True
        
        # Patch BaseChatAgent.on_messages_stream
        if hasattr(BaseChatAgent, "on_messages_stream"):
            if not hasattr(BaseChatAgent.on_messages_stream, "_arzule_patched"):
                _original_methods["on_messages_stream"] = BaseChatAgent.on_messages_stream
                BaseChatAgent.on_messages_stream = _patched_on_messages_stream(
                    BaseChatAgent.on_messages_stream
                )
                BaseChatAgent.on_messages_stream._arzule_patched = True
        
        installed.append("agent_lifecycle")
    
    # Patch model client
    if enable_llm_hooks:
        try:
            from autogen_core.models import ChatCompletionClient
            
            if hasattr(ChatCompletionClient, "create"):
                if not hasattr(ChatCompletionClient.create, "_arzule_patched"):
                    _original_methods["model_create"] = ChatCompletionClient.create
                    ChatCompletionClient.create = _patched_create(ChatCompletionClient.create)
                    ChatCompletionClient.create._arzule_patched = True
            
            installed.append("model_client")
        except ImportError:
            print("[arzule] autogen-core models not available", file=sys.stderr)
    
    # Patch tool execution
    if enable_tool_hooks:
        try:
            from autogen_core.tools import BaseTool
            
            if hasattr(BaseTool, "run"):
                if not hasattr(BaseTool.run, "_arzule_patched"):
                    _original_methods["tool_run"] = BaseTool.run
                    BaseTool.run = _patched_run_tool(BaseTool.run)
                    BaseTool.run._arzule_patched = True
            
            installed.append("tools")
        except ImportError:
            print("[arzule] autogen-core tools not available", file=sys.stderr)
    
    # Hook into autogen-core telemetry if available
    if enable_telemetry:
        try:
            from .telemetry import install_telemetry_hooks
            install_telemetry_hooks()
            installed.append("telemetry")
        except Exception as e:
            print(f"[arzule] Could not install telemetry hooks: {e}", file=sys.stderr)
    
    print(f"[arzule] AutoGen v0.7+ hooks installed: {', '.join(installed)}", file=sys.stderr)


def uninstall_hooks() -> None:
    """Remove monkey-patching hooks (restore original methods)."""
    global _original_methods
    
    try:
        from autogen_agentchat.agents import BaseChatAgent
        from autogen_core.models import ChatCompletionClient
        from autogen_core.tools import BaseTool
    except ImportError:
        return
    
    for method_name, original in _original_methods.items():
        if method_name == "on_messages" and hasattr(BaseChatAgent, "on_messages"):
            BaseChatAgent.on_messages = original
        elif method_name == "on_messages_stream" and hasattr(BaseChatAgent, "on_messages_stream"):
            BaseChatAgent.on_messages_stream = original
        elif method_name == "model_create" and hasattr(ChatCompletionClient, "create"):
            ChatCompletionClient.create = original
        elif method_name == "tool_run" and hasattr(BaseTool, "run"):
            BaseTool.run = original
    
    _original_methods.clear()
    print("[arzule] AutoGen v0.7+ hooks uninstalled", file=sys.stderr)












