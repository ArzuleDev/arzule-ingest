"""Hook integration for Claude Code validators.

This module provides the ValidatorHooks class that integrates with
Claude Code hooks to enable parallel out-of-band validation:

1. PreToolUse (Task) - Assess risk and optionally spawn validators
2. SubagentStop - Retrieve validation results after subagent completes

The hooks are designed to be non-blocking and fault-tolerant - validation
failures should not prevent the main agent from functioning.

Usage in .claude/hooks/arzule-validators.py:
    #!/usr/bin/env python3
    import json
    import sys
    from arzule_ingest.validators import ValidatorsClient, ValidatorHooks

    input_data = json.load(sys.stdin)
    client = ValidatorsClient()
    hooks = ValidatorHooks(client)

    result = hooks.handle_event(input_data)
    if result:
        print(json.dumps(result))
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from .client import ValidatorsClient, ValidatorsClientError
from .risk_gate import DeterministicRiskGate, RiskAssessment, ValidationLevel
from .types import (
    HookContext,
    SpawnRequest,
    SpawnResponse,
    ValidationResult,
    ValidationDecision,
)


# State file for persisting context across hook invocations
_STATE_DIR = Path.home() / ".arzule" / "validators"


class ValidatorHooks:
    """Hook handlers for Claude Code validator integration.

    This class provides methods to handle PreToolUse and SubagentStop
    hooks for parallel out-of-band validation of Claude Code subagents.

    The validation flow is:
    1. PreToolUse (Task tool) fires -> assess risk -> spawn validators if needed
    2. Subagent executes its task
    3. SubagentStop fires -> retrieve validation results -> optionally block

    Attributes:
        client: ValidatorsClient for API communication.
        risk_gate: DeterministicRiskGate for risk assessment.
        project_id: Arzule project ID.
        enabled: Whether validation is enabled.
        debug: Enable debug logging.
    """

    def __init__(
        self,
        client: Optional[ValidatorsClient] = None,
        project_id: Optional[str] = None,
        enabled: Optional[bool] = None,
        strict_mode: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize validator hooks.

        Args:
            client: ValidatorsClient instance (creates default if None).
            project_id: Override for project ID.
            enabled: Enable/disable validation (default: from env).
            strict_mode: Enable strict risk assessment mode.
            debug: Enable debug logging.
        """
        self.client = client or ValidatorsClient(debug=debug)
        self.project_id = (
            project_id
            or os.environ.get("ARZULE_PROJECT_ID")
            or ""
        )
        self.debug = debug

        # Check if validation is enabled
        if enabled is not None:
            self.enabled = enabled
        else:
            # Default: enabled if ARZULE_VALIDATORS_ENABLED=true or api key exists
            env_enabled = os.environ.get("ARZULE_VALIDATORS_ENABLED", "").lower()
            self.enabled = env_enabled in ("true", "1", "yes") or bool(
                os.environ.get("ARZULE_API_KEY")
            )

        # Initialize risk gate
        self.risk_gate = DeterministicRiskGate(
            config={"strict_mode": strict_mode}
        )

        # Ensure state directory exists
        _STATE_DIR.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str, always: bool = False) -> None:
        """Log message to ~/.arzule/hook_debug.log.

        Args:
            message: Message to log.
            always: If True, log even if debug mode is disabled.
        """
        if not self.debug and not always:
            return
        try:
            from datetime import datetime, timezone
            log_file = Path.home() / ".arzule" / "hook_debug.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).isoformat()
            with open(log_file, "a") as f:
                f.write(f"[{timestamp}] [VALIDATOR] {message}\n")
        except Exception:
            pass

    def _get_state_path(self, session_id: str) -> Path:
        """Get path to state file for a session."""
        return _STATE_DIR / f"{session_id}.json"

    def _load_state(self, session_id: str) -> dict[str, Any]:
        """Load persisted state for a session."""
        state_path = self._get_state_path(session_id)
        if state_path.exists():
            try:
                return json.loads(state_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_state(self, session_id: str, state: dict[str, Any]) -> None:
        """Persist state for a session."""
        state_path = self._get_state_path(session_id)
        try:
            state_path.write_text(json.dumps(state, indent=2))
        except OSError as e:
            self._log(f"Failed to save state: {e}")

    def _update_state(self, session_id: str, updates: dict[str, Any]) -> None:
        """Update persisted state for a session."""
        state = self._load_state(session_id)
        state.update(updates)
        self._save_state(session_id, state)

    def handle_event(self, input_data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Route hook event to appropriate handler.

        This is the main entry point for hook scripts. It examines the
        hook_event_name and routes to the appropriate handler.

        Args:
            input_data: Hook input data from Claude Code (stdin JSON).

        Returns:
            Hook response dict, or None for no response.

        Example:
            >>> hooks = ValidatorHooks()
            >>> result = hooks.handle_event(json.load(sys.stdin))
            >>> if result:
            ...     print(json.dumps(result))
        """
        if not self.enabled:
            return None

        event_name = input_data.get("hook_event_name")

        if event_name == "PreToolUse":
            tool_name = input_data.get("tool_name")
            if tool_name == "Task":
                return self.handle_pre_task(input_data)

        elif event_name == "SubagentStop":
            return self.handle_subagent_stop(input_data)

        return None

    def handle_pre_task(
        self, input_data: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Handle PreToolUse hook for Task tool.

        This handler:
        1. Extracts subagent info from tool_input
        2. Runs deterministic risk assessment
        3. Spawns validators if risk warrants validation
        4. Stores context for SubagentStop correlation

        Args:
            input_data: PreToolUse hook input.

        Returns:
            Hook response (usually None to allow Task to proceed).
        """
        session_id = input_data.get("session_id", "")
        tool_use_id = input_data.get("tool_use_id", "")
        tool_input = input_data.get("tool_input", {})

        # Extract subagent details
        subagent_type = tool_input.get("subagent_type", "general-purpose")
        description = tool_input.get("description", "")
        prompt = tool_input.get("prompt", "")

        self._log(f"━━━ Checking subagent: {subagent_type} ━━━", always=True)
        self._log(f"    Description: {description[:80]}{'...' if len(description) > 80 else ''}", always=True)

        # Load session context for historical risk factors
        session_state = self._load_state(session_id)
        session_context = {
            "previous_failure": session_state.get("previous_failure", False),
            "subagent_count": session_state.get("subagent_count", 0),
            "recent_validation_failures": session_state.get(
                "recent_validation_failures", 0
            ),
        }

        # Run risk assessment
        assessment = self.risk_gate.assess(
            subagent_type=subagent_type,
            description=description,
            prompt=prompt,
            session_context=session_context,
        )

        self._log(f"    Risk Score: {assessment.total_score}/100 → Level: {assessment.level.value.upper()}", always=True)
        if assessment.recommended_validators:
            self._log(f"    Validators to run: {', '.join(assessment.recommended_validators)}", always=True)
        if assessment.triggered_signals:
            self._log(f"    Risk signals: {', '.join(assessment.triggered_signals[:5])}", always=True)

        # Store assessment info for SubagentStop
        pending_validations = session_state.get("pending_validations", {})
        pending_validations[tool_use_id] = {
            "subagent_type": subagent_type,
            "description": description,
            "prompt": prompt,
            "risk_score": assessment.total_score,
            "risk_level": assessment.level.value,
            "validators": assessment.recommended_validators,
            "spawned": False,
        }

        # Update subagent count
        session_state["subagent_count"] = session_state.get("subagent_count", 0) + 1
        session_state["pending_validations"] = pending_validations
        self._save_state(session_id, session_state)

        # Skip validation if risk level is SKIP
        if assessment.level == ValidationLevel.SKIP:
            self._log(f"    ✓ Skipping validation (low risk)", always=True)
            return None

        # NOTE: We don't spawn validators here because we don't have agent_id yet.
        # agent_id comes in SubagentStart or SubagentStop, not PreToolUse.
        # We'll spawn validators in SubagentStop if validation is warranted.

        # For now, just return None to allow Task to proceed
        return None

    def handle_subagent_stop(
        self, input_data: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Handle SubagentStop hook.

        This handler:
        1. Correlates with pending validation from PreToolUse
        2. Spawns validators if not already spawned
        3. Retrieves validation results
        4. Returns block decision if critical issues found

        Args:
            input_data: SubagentStop hook input.

        Returns:
            Hook response with continue/block decision.
        """
        session_id = input_data.get("session_id", "")
        agent_id = input_data.get("agent_id", "")
        agent_transcript_path = input_data.get("agent_transcript_path", "")

        self._log(f"━━━ Subagent completed, checking validation ━━━", always=True)

        # Load session state
        session_state = self._load_state(session_id)
        pending_validations = session_state.get("pending_validations", {})

        # Try to find the corresponding pending validation
        # Match by finding the most recent unprocessed validation
        matched_tool_use_id = None
        matched_validation = None

        for tool_use_id, validation in pending_validations.items():
            if not validation.get("processed"):
                matched_tool_use_id = tool_use_id
                matched_validation = validation
                break

        if not matched_validation:
            self._log("    No pending validation for this subagent", always=True)
            return {"continue": True}

        subagent_type = matched_validation.get("subagent_type", "unknown")
        self._log(f"    Subagent type: {subagent_type}", always=True)
        self._log(f"    Risk level: {matched_validation.get('risk_level', 'unknown').upper()}", always=True)

        # Check if validation should be skipped
        risk_level = matched_validation.get("risk_level", "skip")
        if risk_level == "skip":
            self._log("    ✓ Validation skipped (low risk)", always=True)
            matched_validation["processed"] = True
            self._save_state(session_id, session_state)
            return {"continue": True}

        # Spawn validators now that we have agent_id
        validators = matched_validation.get("validators", [])
        if not validators:
            self._log("    ✓ No validators configured", always=True)
            matched_validation["processed"] = True
            self._save_state(session_id, session_state)
            return {"continue": True}

        try:
            # Build spawn request
            request = SpawnRequest(
                project_id=self.project_id,
                session_id=session_id,
                agent_id=agent_id,
                subagent_type=matched_validation.get("subagent_type", "unknown"),
                description=matched_validation.get("description", ""),
                prompt=matched_validation.get("prompt", ""),
                transcript_path=agent_transcript_path,
                validators=validators,
                validation_level=self._level_to_validation_level(risk_level),
                metadata={
                    "tool_use_id": matched_tool_use_id,
                    "risk_score": matched_validation.get("risk_score", 0),
                },
            )

            # Spawn validators
            self._log(f"    ⏳ Spawning validators: {', '.join(validators)}...", always=True)
            spawn_response = self.client.spawn_validators(request)
            self._log(f"    Validators spawned: {len(spawn_response.validator_ids)} running", always=True)

            # Wait for results (with reasonable timeout)
            self._log(f"    ⏳ Waiting for validation results...", always=True)
            result = self.client.get_results(
                session_id=session_id,
                agent_id=agent_id,
                wait_timeout_ms=15000,  # 15 second wait (validators take ~10-13s)
            )

            if result:
                decision_icon = "✓" if result.decision.value == "continue" else "✗" if result.decision.value == "block" else "⚠"
                self._log(f"    {decision_icon} Validation Result: {result.decision.value.upper()}", always=True)
                self._log(f"      Score: {result.aggregate_score:.2f} | Completed: {result.validators_completed}/{result.validators_total}", always=True)
                if result.critical_issues:
                    self._log(f"      ⚠ Critical issues: {len(result.critical_issues)}", always=True)
                if result.recommendations:
                    self._log(f"      Recommendations: {len(result.recommendations)}", always=True)

                # Mark as processed
                matched_validation["processed"] = True
                matched_validation["result"] = {
                    "decision": result.decision.value,
                    "score": result.aggregate_score,
                    "critical_count": len(result.critical_issues),
                    "high_count": len(result.high_issues),
                }

                # Update failure tracking
                if result.should_block():
                    session_state["recent_validation_failures"] = (
                        session_state.get("recent_validation_failures", 0) + 1
                    )
                    session_state["previous_failure"] = True
                else:
                    session_state["previous_failure"] = False

                self._save_state(session_id, session_state)

                # Return block decision if critical issues found
                if result.should_block():
                    summary = result.get_summary()
                    return {
                        "decision": "block",
                        "reason": summary,
                        "continue": False,
                    }

                return {"continue": True}

            else:
                self._log("No validation results available", always=True)
                matched_validation["processed"] = True
                self._save_state(session_id, session_state)
                return {"continue": True}

        except ValidatorsClientError as e:
            self._log(f"Validation error (non-blocking): {e}", always=True)
            # Mark as processed to avoid retrying
            matched_validation["processed"] = True
            matched_validation["error"] = str(e)
            self._save_state(session_id, session_state)
            # Don't block on validation errors
            return {"continue": True}

    def _level_to_validation_level(self, level: str) -> str:
        """Convert risk level to validation_level parameter."""
        mapping = {
            "lite": "minimal",
            "standard": "standard",
            "deep": "thorough",
        }
        return mapping.get(level, "standard")

    def assess_risk(
        self,
        subagent_type: str,
        description: str,
        prompt: str,
        session_context: Optional[dict[str, Any]] = None,
    ) -> RiskAssessment:
        """Assess risk for a subagent call (public API).

        This method exposes the risk gate for external use without
        triggering any API calls or state management.

        Args:
            subagent_type: Type of subagent.
            description: Task description.
            prompt: Full prompt text.
            session_context: Optional session context for historical factors.

        Returns:
            RiskAssessment with score and recommendations.
        """
        return self.risk_gate.assess(
            subagent_type=subagent_type,
            description=description,
            prompt=prompt,
            session_context=session_context,
        )


def create_hook_script() -> str:
    """Generate the content for a Claude Code hook script.

    Returns:
        Python script content for .claude/hooks/arzule-validators.py
    """
    return '''#!/usr/bin/env python3
"""Arzule Validators Hook for Claude Code"""
import json
import sys

def main():
    try:
        from arzule_ingest.validators import ValidatorsClient, ValidatorHooks
    except ImportError:
        sys.exit(0)
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)
    try:
        client = ValidatorsClient()
        hooks = ValidatorHooks(client)
        result = hooks.handle_event(input_data)
        if result:
            print(json.dumps(result))
    except Exception as e:
        print(f"Arzule validators hook error: {e}", file=sys.stderr)
    sys.exit(0)

if __name__ == "__main__":
    main()
'''


def main():
    """CLI entry point for the validators hook."""
    import json
    import sys

    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    try:
        client = ValidatorsClient()
        hooks = ValidatorHooks(client)
        result = hooks.handle_event(input_data)

        if result:
            print(json.dumps(result))

    except Exception as e:
        # Log error but don't block Claude Code
        print(f"Arzule validators hook error: {e}", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
