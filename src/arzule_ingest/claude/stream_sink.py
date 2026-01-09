"""Real-time streaming sink for Claude Code events.

Sends events to a local observability server (like the reference repo)
in addition to the Arzule backend. This enables real-time monitoring
while still collecting data for analysis.

To enable, set ARZULE_STREAM_URL=http://localhost:4000/events
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from datetime import datetime
from typing import Any


class StreamingSink:
    """
    Sink that streams events to a local observability server.
    
    Compatible with the reference repo's server format:
    POST /events with {source_app, session_id, hook_event_type, payload}
    """
    
    def __init__(
        self,
        server_url: str = "http://localhost:4000/events",
        session_id: str = "",
        source_app: str = "arzule-claude",
        timeout: float = 2.0,
    ):
        """
        Initialize the streaming sink.
        
        Args:
            server_url: URL of the observability server
            session_id: Claude Code session ID
            source_app: Application name for event grouping
            timeout: Request timeout in seconds
        """
        self.server_url = server_url
        self.session_id = session_id
        self.source_app = source_app
        self.timeout = timeout
    
    def emit(self, event: dict) -> None:
        """
        Stream a single event to the observability server.
        
        Converts from TraceEvent format to the simpler streaming format.
        """
        # Convert TraceEvent to streaming format
        stream_event = self._convert_event(event)
        
        try:
            self._send_event(stream_event)
        except Exception:
            # Fail silently - streaming is best-effort
            pass
    
    def flush(self) -> None:
        """Flush is a no-op for streaming (events sent immediately)."""
        pass
    
    def close(self) -> None:
        """Close is a no-op for streaming."""
        pass
    
    def _convert_event(self, trace_event: dict) -> dict:
        """
        Convert TraceEvent to streaming format.
        
        TraceEvent format:
        {
            run_id, tenant_id, project_id, trace_id, span_id,
            event_type, agent, summary, attrs_compact, payload, ...
        }
        
        Streaming format (reference repo compatible):
        {
            source_app, session_id, hook_event_type, payload,
            timestamp, summary?, model_name?
        }
        """
        event_type = trace_event.get("event_type", "unknown")
        
        # Map event_type to hook_event_type
        hook_type_map = {
            "session.start": "SessionStart",
            "session.end": "SessionEnd",
            "turn.start": "UserPromptSubmit",
            "turn.end": "Stop",
            "user.prompt": "UserPromptSubmit",
            "tool.call.start": "PreToolUse",
            "tool.call.end": "PostToolUse",
            "tool.call.blocked": "PreToolUse",
            "tool.call.warning": "PreToolUse",
            "handoff.proposed": "PreToolUse",
            "handoff.ack": "PreToolUse",
            "handoff.complete": "PostToolUse",
            "agent.response.complete": "Stop",
            "agent.end": "SubagentStop",
            "context.compact": "PreCompact",
            "notification": "Notification",
        }
        hook_event_type = hook_type_map.get(event_type, event_type)
        
        # Build payload from attrs_compact and payload
        payload = {}
        if trace_event.get("attrs_compact"):
            payload.update(trace_event["attrs_compact"])
        if trace_event.get("payload"):
            payload.update(trace_event["payload"])
        
        # Add agent info if present
        agent = trace_event.get("agent")
        if agent:
            payload["agent_id"] = agent.get("id")
            payload["agent_role"] = agent.get("role")
        
        return {
            "source_app": self.source_app,
            "session_id": self.session_id,
            "hook_event_type": hook_event_type,
            "payload": payload,
            "timestamp": int(datetime.now().timestamp() * 1000),
            "summary": trace_event.get("summary"),
        }
    
    def _send_event(self, event_data: dict) -> bool:
        """Send event to the observability server."""
        try:
            req = urllib.request.Request(
                self.server_url,
                data=json.dumps(event_data).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Arzule-Claude-Hook/1.0",
                },
            )
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return response.status == 200
                
        except urllib.error.URLError:
            return False
        except Exception:
            return False



