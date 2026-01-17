"""Arzule Validators - Parallel out-of-band validation for Claude Code subagents.

This module provides SDK integration with the Arzule validators backend service.
It enables validation of Claude Code subagents through:

1. Deterministic Risk Gate - Fast (<1ms) rule-based risk assessment
2. PreToolUse hook - Spawns validators when Task tool is called
3. SubagentStop hook - Retrieves validation results after subagent completes

Components:

- `DeterministicRiskGate`: Fast rule-based risk scoring
- `ValidatorsClient`: API client for validators backend
- `ValidatorHooks`: Claude Code hook handlers
- `ValidationStreamClient`: Streaming results support

Usage in Claude Code hooks:

    from arzule_ingest.validators import ValidatorsClient, ValidatorHooks

    # Create client and hooks handler
    client = ValidatorsClient()
    hooks = ValidatorHooks(client)

    # In hook script (stdin receives JSON)
    import json, sys
    input_data = json.load(sys.stdin)
    result = hooks.handle_event(input_data)
    if result:
        print(json.dumps(result))

Risk assessment without API calls:

    from arzule_ingest.validators import should_validate

    should_val, assessment = should_validate(
        subagent_type="general-purpose",
        description="Update database schema",
        prompt="Modify the users table to add email column",
    )

    if should_val:
        print(f"Validate at level: {assessment.level.value}")
        print(f"Validators: {assessment.recommended_validators}")
"""

from __future__ import annotations

# Risk gate (Tier 0 - deterministic)
from .risk_gate import (
    DeterministicRiskGate,
    RiskAssessment,
    ValidationLevel,
    should_validate,
)

# API client
from .client import (
    ValidatorsClient,
    ValidatorsClientError,
    AuthenticationError,
    ValidationError,
    NetworkError,
)

# Hook handlers
from .hooks import (
    ValidatorHooks,
    create_hook_script,
)

# Type definitions
from .types import (
    SpawnRequest,
    SpawnResponse,
    ValidationResult,
    ValidatorResult,
    ValidatorIssue,
    ValidationDecision,
    Severity,
    HookContext,
)

# Streaming support
from .streaming import (
    ValidationStreamClient,
    ProgressiveResultCollector,
    StreamEvent,
    stream_validation_progress,
)

# Event mapping utilities
from .event_mapping import (
    TaskToolInput,
    extract_pre_tool_use_context,
    extract_subagent_stop_context,
    build_spawn_request,
    map_result_to_hook_response,
    correlate_subagent_to_task,
)

__all__ = [
    # Risk gate
    "DeterministicRiskGate",
    "RiskAssessment",
    "ValidationLevel",
    "should_validate",
    # Client
    "ValidatorsClient",
    "ValidatorsClientError",
    "AuthenticationError",
    "ValidationError",
    "NetworkError",
    # Hooks
    "ValidatorHooks",
    "create_hook_script",
    # Types
    "SpawnRequest",
    "SpawnResponse",
    "ValidationResult",
    "ValidatorResult",
    "ValidatorIssue",
    "ValidationDecision",
    "Severity",
    "HookContext",
    # Streaming
    "ValidationStreamClient",
    "ProgressiveResultCollector",
    "StreamEvent",
    "stream_validation_progress",
    # Event mapping
    "TaskToolInput",
    "extract_pre_tool_use_context",
    "extract_subagent_stop_context",
    "build_spawn_request",
    "map_result_to_hook_response",
    "correlate_subagent_to_task",
]
