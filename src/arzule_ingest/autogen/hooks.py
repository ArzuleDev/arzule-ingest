"""Monkey-patching hooks for AutoGen instrumentation.

This module patches AutoGen's ConversableAgent class methods to capture:
- Message send/receive events
- LLM calls
- Tool/function executions
- Code executions
- Conversation lifecycle
"""

from __future__ import annotations

import functools
import sys
from typing import Any, Callable, Optional, TYPE_CHECKING

from ..ids import new_span_id
from ..run import current_run
from .normalize import (
    evt_message_send,
    evt_message_receive,
    evt_llm_start,
    evt_llm_end,
    evt_tool_start,
    evt_tool_end,
    evt_code_execution,
    evt_conversation_start,
    evt_conversation_end,
)
from .handoff import detect_handoff, emit_handoff_event
from .spanctx import (
    start_conversation_span,
    end_conversation_span,
    get_message_span_id,
)

if TYPE_CHECKING:
    pass

# Store original methods for restoration
_original_methods: dict[str, Any] = {}

# Track conversation message counts
_conversation_message_counts: dict[str, int] = {}


def _get_agent_name(agent: Any) -> str:
    """Safely get agent name."""
    try:
        return getattr(agent, "name", None) or "unknown"
    except Exception:
        return "unknown"


# =============================================================================
# Send Hook
# =============================================================================


def _patched_send(
    original_send: Callable,
) -> Callable:
    """Create patched send method."""
    
    @functools.wraps(original_send)
    def wrapper(
        self,
        message: Any,
        recipient: Any,
        request_reply: Optional[bool] = None,
        silent: bool = False,
        **kwargs,
    ):
        run = current_run()
        
        if run:
            try:
                # Generate span ID for this message
                span_id = get_message_span_id()
                
                # Emit send event
                evt = evt_message_send(
                    run=run,
                    sender=self,
                    recipient=recipient,
                    message=message,
                    span_id=span_id,
                )
                run.emit(evt)
                
                # Detect and emit handoff
                sender_name = _get_agent_name(self)
                recipient_name = _get_agent_name(recipient)
                handoff_key = detect_handoff(sender_name, recipient_name, message)
                
                if handoff_key:
                    # Extract message content for summary
                    content = ""
                    if isinstance(message, str):
                        content = message[:50]
                    elif isinstance(message, dict):
                        content = str(message.get("content", ""))[:50]
                    
                    emit_handoff_event(
                        run=run,
                        from_agent=sender_name,
                        to_agent=recipient_name,
                        handoff_key=handoff_key,
                        message_content=content,
                        span_id=span_id,
                    )
                    
            except Exception as e:
                print(f"[arzule] Error in send hook: {e}", file=sys.stderr)
        
        return original_send(self, message, recipient, request_reply, silent, **kwargs)
    
    return wrapper


# =============================================================================
# Receive Hook
# =============================================================================


def _patched_receive(
    original_receive: Callable,
) -> Callable:
    """Create patched receive method."""
    
    @functools.wraps(original_receive)
    def wrapper(
        self,
        message: Any,
        sender: Any,
        request_reply: Optional[bool] = None,
        silent: bool = False,
        **kwargs,
    ):
        run = current_run()
        
        if run:
            try:
                # Generate span ID for this message
                span_id = get_message_span_id()
                
                # Emit receive event
                evt = evt_message_receive(
                    run=run,
                    sender=sender,
                    recipient=self,
                    message=message,
                    request_reply=request_reply,
                    span_id=span_id,
                )
                run.emit(evt)
                
            except Exception as e:
                print(f"[arzule] Error in receive hook: {e}", file=sys.stderr)
        
        return original_receive(self, message, sender, request_reply, silent, **kwargs)
    
    return wrapper


# =============================================================================
# Generate OAI Reply Hook (LLM Calls)
# =============================================================================


def _patched_generate_oai_reply(
    original_generate: Callable,
) -> Callable:
    """Create patched _generate_oai_reply method for LLM call capture."""
    
    @functools.wraps(original_generate)
    def wrapper(self, messages: Optional[list] = None, sender: Any = None, config: Any = None, **kwargs):
        run = current_run()
        
        if not run:
            return original_generate(self, messages, sender, config, **kwargs)
        
        span_id = new_span_id()
        span_pushed = False
        
        try:
            # Emit LLM start event
            start_evt = evt_llm_start(
                run=run,
                agent=self,
                messages=messages or [],
                span_id=span_id,
            )
            run.emit(start_evt)
            run.push_span(span_id)
            span_pushed = True
            
        except Exception as e:
            print(f"[arzule] Error in LLM start hook: {e}", file=sys.stderr)
        
        error = None
        result = None
        
        try:
            result = original_generate(self, messages, sender, config, **kwargs)
            return result
        except Exception as e:
            error = e
            raise
        finally:
            if run:
                try:
                    # Extract response from result if available
                    response = None
                    if result is not None:
                        if isinstance(result, tuple) and len(result) >= 2:
                            # AutoGen returns (success, response) tuple
                            response = result[1]
                        else:
                            response = result
                    
                    end_evt = evt_llm_end(
                        run=run,
                        agent=self,
                        response=response,
                        span_id=span_id,
                        error=error,
                    )
                    run.emit(end_evt)
                    
                    # Only pop if we successfully pushed
                    if span_pushed:
                        run.pop_span()
                    
                except Exception as e:
                    print(f"[arzule] Error in LLM end hook: {e}", file=sys.stderr)
    
    return wrapper


# =============================================================================
# Execute Function Hook (Tool Calls)
# =============================================================================


def _patched_execute_function(
    original_execute: Callable,
) -> Callable:
    """Create patched execute_function method for tool call capture."""
    
    @functools.wraps(original_execute)
    def wrapper(self, func_call: dict, **kwargs):
        run = current_run()
        
        if not run:
            return original_execute(self, func_call, **kwargs)
        
        span_id = new_span_id()
        span_pushed = False
        tool_name = func_call.get("name", "unknown") if isinstance(func_call, dict) else "unknown"
        tool_args = func_call.get("arguments", {}) if isinstance(func_call, dict) else {}
        
        try:
            # Emit tool start event
            start_evt = evt_tool_start(
                run=run,
                agent=self,
                tool_name=tool_name,
                tool_input=tool_args,
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
            result = original_execute(self, func_call, **kwargs)
            return result
        except Exception as e:
            error = e
            raise
        finally:
            if run:
                try:
                    end_evt = evt_tool_end(
                        run=run,
                        agent=self,
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
# Execute Code Blocks Hook
# =============================================================================


def _patched_execute_code_blocks(
    original_execute: Callable,
) -> Callable:
    """Create patched execute_code_blocks method for code execution capture."""
    
    @functools.wraps(original_execute)
    def wrapper(self, code_blocks: list, **kwargs):
        run = current_run()
        
        if not run:
            return original_execute(self, code_blocks, **kwargs)
        
        result = original_execute(self, code_blocks, **kwargs)
        
        try:
            # Result is typically (exit_code, logs, code_filename)
            exit_code = 0
            output = ""
            
            if isinstance(result, tuple):
                if len(result) >= 1:
                    exit_code = result[0] if isinstance(result[0], int) else 0
                if len(result) >= 2:
                    output = str(result[1]) if result[1] else ""
            
            # Combine all code blocks for logging
            code_content = ""
            for block in code_blocks:
                if isinstance(block, tuple) and len(block) >= 2:
                    code_content += block[1] + "\n"
                elif isinstance(block, str):
                    code_content += block + "\n"
            
            evt = evt_code_execution(
                run=run,
                agent=self,
                code=code_content.strip(),
                output=output,
                exit_code=exit_code,
            )
            run.emit(evt)
            
        except Exception as e:
            print(f"[arzule] Error in code execution hook: {e}", file=sys.stderr)
        
        return result
    
    return wrapper


# =============================================================================
# Initiate Chat Hook (Conversation Start/End)
# =============================================================================


def _patched_initiate_chat(
    original_initiate: Callable,
) -> Callable:
    """Create patched initiate_chat method for conversation lifecycle capture."""
    
    @functools.wraps(original_initiate)
    def wrapper(self, recipient: Any, *args, **kwargs):
        run = current_run()
        
        # Extract message for logging (may be positional or keyword)
        message = kwargs.get("message")
        if not message and args:
            # message might be in positional args
            message = args[0] if args else None
        
        if not run:
            return original_initiate(self, recipient, *args, **kwargs)
        
        # Generate conversation ID and span
        import uuid
        conversation_id = str(uuid.uuid4())
        span_id = None
        span_started = False
        
        try:
            span_id = start_conversation_span(run, conversation_id)
            span_started = True
        except Exception as e:
            print(f"[arzule] Error starting conversation span: {e}", file=sys.stderr)
        
        # Track message count
        _conversation_message_counts[conversation_id] = 0
        
        try:
            # Emit conversation start event
            participants = [self, recipient]
            start_evt = evt_conversation_start(
                run=run,
                initiator=self,
                participants=participants,
                initial_message=message,
                span_id=span_id,
            )
            run.emit(start_evt)
            
        except Exception as e:
            print(f"[arzule] Error in conversation start hook: {e}", file=sys.stderr)
        
        error = None
        result = None
        
        try:
            result = original_initiate(self, recipient, *args, **kwargs)
            return result
        except Exception as e:
            error = e
            raise
        finally:
            if run:
                try:
                    # Calculate message count from chat history if available
                    message_count = _conversation_message_counts.get(conversation_id, 0)
                    
                    if result and hasattr(result, "chat_history"):
                        message_count = len(result.chat_history)
                    
                    end_evt = evt_conversation_end(
                        run=run,
                        initiator=self,
                        result=result,
                        message_count=message_count,
                        span_id=span_id,
                        status="error" if error else "ok",
                    )
                    run.emit(end_evt)
                    
                    # Only end the span if we successfully started it
                    if span_started:
                        end_conversation_span(run, conversation_id)
                    
                    # Clean up message count
                    _conversation_message_counts.pop(conversation_id, None)
                    
                except Exception as e:
                    print(f"[arzule] Error in conversation end hook: {e}", file=sys.stderr)
    
    return wrapper


# =============================================================================
# Async Variants
# =============================================================================


def _patched_async_send(
    original_send: Callable,
) -> Callable:
    """Create patched async send method."""
    
    @functools.wraps(original_send)
    async def wrapper(
        self,
        message: Any,
        recipient: Any,
        request_reply: Optional[bool] = None,
        silent: bool = False,
        **kwargs,
    ):
        run = current_run()
        
        if run:
            try:
                span_id = get_message_span_id()
                evt = evt_message_send(
                    run=run,
                    sender=self,
                    recipient=recipient,
                    message=message,
                    span_id=span_id,
                )
                run.emit(evt)
                
                # Detect handoff
                sender_name = _get_agent_name(self)
                recipient_name = _get_agent_name(recipient)
                handoff_key = detect_handoff(sender_name, recipient_name, message)
                
                if handoff_key:
                    content = ""
                    if isinstance(message, str):
                        content = message[:50]
                    elif isinstance(message, dict):
                        content = str(message.get("content", ""))[:50]
                    
                    emit_handoff_event(
                        run=run,
                        from_agent=sender_name,
                        to_agent=recipient_name,
                        handoff_key=handoff_key,
                        message_content=content,
                        span_id=span_id,
                    )
                    
            except Exception as e:
                print(f"[arzule] Error in async send hook: {e}", file=sys.stderr)
        
        return await original_send(self, message, recipient, request_reply, silent, **kwargs)
    
    return wrapper


def _patched_async_receive(
    original_receive: Callable,
) -> Callable:
    """Create patched async receive method."""
    
    @functools.wraps(original_receive)
    async def wrapper(
        self,
        message: Any,
        sender: Any,
        request_reply: Optional[bool] = None,
        silent: bool = False,
        **kwargs,
    ):
        run = current_run()
        
        if run:
            try:
                span_id = get_message_span_id()
                evt = evt_message_receive(
                    run=run,
                    sender=sender,
                    recipient=self,
                    message=message,
                    request_reply=request_reply,
                    span_id=span_id,
                )
                run.emit(evt)
                
            except Exception as e:
                print(f"[arzule] Error in async receive hook: {e}", file=sys.stderr)
        
        return await original_receive(self, message, sender, request_reply, silent, **kwargs)
    
    return wrapper


def _patched_async_initiate_chat(
    original_initiate: Callable,
) -> Callable:
    """Create patched async initiate_chat method."""
    
    @functools.wraps(original_initiate)
    async def wrapper(self, recipient: Any, *args, **kwargs):
        run = current_run()
        
        # Extract message for logging
        message = kwargs.get("message")
        if not message and args:
            message = args[0] if args else None
        
        if not run:
            return await original_initiate(self, recipient, *args, **kwargs)
        
        import uuid
        conversation_id = str(uuid.uuid4())
        span_id = None
        span_started = False
        
        try:
            span_id = start_conversation_span(run, conversation_id)
            span_started = True
        except Exception as e:
            print(f"[arzule] Error starting async conversation span: {e}", file=sys.stderr)
        
        _conversation_message_counts[conversation_id] = 0
        
        try:
            participants = [self, recipient]
            start_evt = evt_conversation_start(
                run=run,
                initiator=self,
                participants=participants,
                initial_message=message,
                span_id=span_id,
            )
            run.emit(start_evt)
            
        except Exception as e:
            print(f"[arzule] Error in async conversation start hook: {e}", file=sys.stderr)
        
        error = None
        result = None
        
        try:
            result = await original_initiate(self, recipient, *args, **kwargs)
            return result
        except Exception as e:
            error = e
            raise
        finally:
            if run:
                try:
                    message_count = _conversation_message_counts.get(conversation_id, 0)
                    
                    if result and hasattr(result, "chat_history"):
                        message_count = len(result.chat_history)
                    
                    end_evt = evt_conversation_end(
                        run=run,
                        initiator=self,
                        result=result,
                        message_count=message_count,
                        span_id=span_id,
                        status="error" if error else "ok",
                    )
                    run.emit(end_evt)
                    
                    # Only end the span if we successfully started it
                    if span_started:
                        end_conversation_span(run, conversation_id)
                    _conversation_message_counts.pop(conversation_id, None)
                    
                except Exception as e:
                    print(f"[arzule] Error in async conversation end hook: {e}", file=sys.stderr)
    
    return wrapper


# =============================================================================
# Installation
# =============================================================================


def install_hooks(
    enable_message_hooks: bool = True,
    enable_llm_hooks: bool = True,
    enable_tool_hooks: bool = True,
    enable_code_execution_hooks: bool = True,
    enable_conversation_hooks: bool = True,
) -> None:
    """
    Install monkey-patching hooks for AutoGen.
    
    Args:
        enable_message_hooks: Install send/receive message hooks
        enable_llm_hooks: Install LLM call hooks
        enable_tool_hooks: Install tool/function execution hooks
        enable_code_execution_hooks: Install code block execution hooks
        enable_conversation_hooks: Install conversation lifecycle hooks
    """
    global _original_methods
    
    try:
        from autogen import ConversableAgent
    except ImportError:
        print("[arzule] AutoGen not installed, skipping hooks", file=sys.stderr)
        return
    
    installed = []
    
    # Store and patch send/receive (message hooks)
    if enable_message_hooks:
        if not hasattr(ConversableAgent.send, "_arzule_patched"):
            _original_methods["send"] = ConversableAgent.send
            ConversableAgent.send = _patched_send(ConversableAgent.send)
            ConversableAgent.send._arzule_patched = True
        
        if not hasattr(ConversableAgent.receive, "_arzule_patched"):
            _original_methods["receive"] = ConversableAgent.receive
            ConversableAgent.receive = _patched_receive(ConversableAgent.receive)
            ConversableAgent.receive._arzule_patched = True
        
        # Async variants
        if hasattr(ConversableAgent, "a_send"):
            if not hasattr(ConversableAgent.a_send, "_arzule_patched"):
                _original_methods["a_send"] = ConversableAgent.a_send
                ConversableAgent.a_send = _patched_async_send(ConversableAgent.a_send)
                ConversableAgent.a_send._arzule_patched = True
        
        if hasattr(ConversableAgent, "a_receive"):
            if not hasattr(ConversableAgent.a_receive, "_arzule_patched"):
                _original_methods["a_receive"] = ConversableAgent.a_receive
                ConversableAgent.a_receive = _patched_async_receive(ConversableAgent.a_receive)
                ConversableAgent.a_receive._arzule_patched = True
        
        installed.append("message")
    
    # Store and patch _generate_oai_reply (LLM hooks)
    if enable_llm_hooks:
        if hasattr(ConversableAgent, "_generate_oai_reply"):
            if not hasattr(ConversableAgent._generate_oai_reply, "_arzule_patched"):
                _original_methods["_generate_oai_reply"] = ConversableAgent._generate_oai_reply
                ConversableAgent._generate_oai_reply = _patched_generate_oai_reply(
                    ConversableAgent._generate_oai_reply
                )
                ConversableAgent._generate_oai_reply._arzule_patched = True
        installed.append("llm")
    
    # Store and patch execute_function (tool hooks)
    if enable_tool_hooks:
        if hasattr(ConversableAgent, "execute_function"):
            if not hasattr(ConversableAgent.execute_function, "_arzule_patched"):
                _original_methods["execute_function"] = ConversableAgent.execute_function
                ConversableAgent.execute_function = _patched_execute_function(
                    ConversableAgent.execute_function
                )
                ConversableAgent.execute_function._arzule_patched = True
        installed.append("tool")
    
    # Store and patch execute_code_blocks (code execution hooks)
    if enable_code_execution_hooks:
        if hasattr(ConversableAgent, "execute_code_blocks"):
            if not hasattr(ConversableAgent.execute_code_blocks, "_arzule_patched"):
                _original_methods["execute_code_blocks"] = ConversableAgent.execute_code_blocks
                ConversableAgent.execute_code_blocks = _patched_execute_code_blocks(
                    ConversableAgent.execute_code_blocks
                )
                ConversableAgent.execute_code_blocks._arzule_patched = True
        installed.append("code")
    
    # Store and patch initiate_chat (conversation hooks)
    if enable_conversation_hooks:
        if not hasattr(ConversableAgent.initiate_chat, "_arzule_patched"):
            _original_methods["initiate_chat"] = ConversableAgent.initiate_chat
            ConversableAgent.initiate_chat = _patched_initiate_chat(ConversableAgent.initiate_chat)
            ConversableAgent.initiate_chat._arzule_patched = True
        
        if hasattr(ConversableAgent, "a_initiate_chat"):
            if not hasattr(ConversableAgent.a_initiate_chat, "_arzule_patched"):
                _original_methods["a_initiate_chat"] = ConversableAgent.a_initiate_chat
                ConversableAgent.a_initiate_chat = _patched_async_initiate_chat(
                    ConversableAgent.a_initiate_chat
                )
                ConversableAgent.a_initiate_chat._arzule_patched = True
        installed.append("conversation")
    
    print(f"[arzule] AutoGen hooks installed: {', '.join(installed)}", file=sys.stderr)


def uninstall_hooks() -> None:
    """Remove monkey-patching hooks (restore original methods)."""
    global _original_methods
    
    try:
        from autogen import ConversableAgent
    except ImportError:
        return
    
    for method_name, original in _original_methods.items():
        if hasattr(ConversableAgent, method_name):
            setattr(ConversableAgent, method_name, original)
    
    _original_methods.clear()
    print("[arzule] AutoGen hooks uninstalled", file=sys.stderr)

