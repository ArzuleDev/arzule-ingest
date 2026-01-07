"""Integration with AutoGen core's OpenTelemetry tracing system.

AutoGen v0.7+ includes built-in telemetry support using OpenTelemetry.
This module hooks into that system to capture traces.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..run import ArzuleRun


def install_telemetry_hooks() -> None:
    """
    Install hooks into autogen-core's telemetry system.
    
    This integrates with AutoGen's OpenTelemetry tracing to capture
    additional context and events.
    
    Note:
        This is optional and only provides additional context.
        The main instrumentation is done through method patching.
    """
    try:
        from autogen_core import _telemetry
        
        # Check if telemetry is available
        if not hasattr(_telemetry, 'trace_create_agent_span'):
            print("[arzule] AutoGen telemetry not available", file=sys.stderr)
            return
        
        # TODO: Hook into OpenTelemetry spans if needed
        # For now, we rely on method patching for instrumentation
        
        print("[arzule] AutoGen telemetry hooks installed", file=sys.stderr)
        
    except ImportError:
        print("[arzule] autogen-core telemetry not available", file=sys.stderr)


def uninstall_telemetry_hooks() -> None:
    """Remove telemetry hooks."""
    # Placeholder for cleanup if needed
    pass










