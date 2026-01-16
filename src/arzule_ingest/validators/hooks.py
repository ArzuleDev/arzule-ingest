"""Hook interface for validator lifecycle events."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from .types import ValidationLevel

if TYPE_CHECKING:
    from .client import ValidatorClient

# Type alias for validator callbacks
ValidatorCallback = Callable[[Dict[str, Any]], None]

# Module-level hooks instance
_hooks_instance: Optional["ValidatorHooksIntegration"] = None


class ValidatorHook(ABC):
    """Abstract base class for validator hooks.

    Implement this class to receive notifications when validators are spawned
    and when validation results arrive. This allows integrating validation
    into custom workflows, logging systems, or UI updates.

    Example:
        class MyValidatorHook(ValidatorHook):
            def on_spawn(self, event, config):
                print(f"Validators spawned for event: {event['event_type']}")

            def on_result(self, event, result):
                if not result.get('passed'):
                    alert_team(result)
    """

    @abstractmethod
    def on_spawn(self, event: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Called when validators are spawned for an event.

        Args:
            event: The event data that triggered validator spawning.
                Contains keys like 'event_type', 'session_id', 'agent_id', etc.
            config: Configuration used for this validation run.
                Contains keys like 'validators', 'level', 'timeout_ms', etc.
        """
        pass

    @abstractmethod
    def on_result(self, event: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Called when validation results arrive.

        Args:
            event: The original event that was validated.
            result: The validation result containing:
                - validator_type: Type of validator that produced this result
                - score: Float from 0.0 to 1.0
                - passed: Boolean indicating if validation passed
                - issues: List of ValidationIssue dicts
                - recommendations: List of recommendation strings
                - metadata: Additional validator-specific data
        """
        pass


class DefaultValidatorHook(ValidatorHook):
    """Default implementation that calls registered callbacks.

    This hook allows registering multiple callbacks for spawn and result events,
    making it easy to integrate validation into existing systems without
    creating custom hook classes.

    Example:
        hook = DefaultValidatorHook()

        def log_spawn(data):
            logger.info(f"Validators spawned: {data}")

        def check_result(data):
            if not data['result'].get('passed'):
                send_alert(data)

        hook.register_spawn_callback(log_spawn)
        hook.register_result_callback(check_result)
    """

    def __init__(self) -> None:
        """Initialize the default validator hook."""
        self._spawn_callbacks: List[ValidatorCallback] = []
        self._result_callbacks: List[ValidatorCallback] = []
        self._error_handler: Optional[Callable[[Exception, str], None]] = None

    def register_spawn_callback(self, callback: ValidatorCallback) -> None:
        """Register a callback to be called when validators are spawned.

        Args:
            callback: Function that accepts a dict with 'event' and 'config' keys.
        """
        if callback not in self._spawn_callbacks:
            self._spawn_callbacks.append(callback)

    def unregister_spawn_callback(self, callback: ValidatorCallback) -> None:
        """Unregister a previously registered spawn callback.

        Args:
            callback: The callback function to remove.
        """
        if callback in self._spawn_callbacks:
            self._spawn_callbacks.remove(callback)

    def register_result_callback(self, callback: ValidatorCallback) -> None:
        """Register a callback to be called when validation results arrive.

        Args:
            callback: Function that accepts a dict with 'event' and 'result' keys.
        """
        if callback not in self._result_callbacks:
            self._result_callbacks.append(callback)

    def unregister_result_callback(self, callback: ValidatorCallback) -> None:
        """Unregister a previously registered result callback.

        Args:
            callback: The callback function to remove.
        """
        if callback in self._result_callbacks:
            self._result_callbacks.remove(callback)

    def set_error_handler(
        self, handler: Optional[Callable[[Exception, str], None]]
    ) -> None:
        """Set a custom error handler for callback exceptions.

        Args:
            handler: Function that accepts (exception, context_string).
                If None, exceptions will be silently ignored.
        """
        self._error_handler = handler

    def on_spawn(self, event: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Call all registered spawn callbacks.

        Args:
            event: The event data that triggered validator spawning.
            config: Configuration used for this validation run.
        """
        callback_data = {"event": event, "config": config}
        for callback in self._spawn_callbacks:
            try:
                callback(callback_data)
            except Exception as e:
                self._handle_error(e, f"spawn callback {callback.__name__}")

    def on_result(self, event: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Call all registered result callbacks.

        Args:
            event: The original event that was validated.
            result: The validation result.
        """
        callback_data = {"event": event, "result": result}
        for callback in self._result_callbacks:
            try:
                callback(callback_data)
            except Exception as e:
                self._handle_error(e, f"result callback {callback.__name__}")

    def _handle_error(self, error: Exception, context: str) -> None:
        """Handle an error from a callback.

        Args:
            error: The exception that was raised.
            context: Description of where the error occurred.
        """
        if self._error_handler:
            try:
                self._error_handler(error, context)
            except Exception:
                # Error handler itself failed, nothing we can do
                pass

    def clear_callbacks(self) -> None:
        """Remove all registered callbacks."""
        self._spawn_callbacks.clear()
        self._result_callbacks.clear()


class LoggingValidatorHook(ValidatorHook):
    """Validator hook that logs all events using Python's logging module.

    Useful for debugging and audit trails.

    Example:
        import logging
        logging.basicConfig(level=logging.INFO)

        hook = LoggingValidatorHook(logger_name="arzule.validators")
    """

    def __init__(self, logger_name: str = "arzule.validators") -> None:
        """Initialize the logging hook.

        Args:
            logger_name: Name of the logger to use.
        """
        import logging

        self._logger = logging.getLogger(logger_name)

    def on_spawn(self, event: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Log validator spawn event.

        Args:
            event: The event data that triggered validator spawning.
            config: Configuration used for this validation run.
        """
        event_type = event.get("event_type", "unknown")
        validators = config.get("validators", [])
        level = config.get("level", "standard")
        self._logger.info(
            "Validators spawned: event_type=%s, validators=%s, level=%s",
            event_type,
            validators,
            level,
        )

    def on_result(self, event: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Log validation result.

        Args:
            event: The original event that was validated.
            result: The validation result.
        """
        event_type = event.get("event_type", "unknown")
        validator_type = result.get("validator_type", "unknown")
        passed = result.get("passed", False)
        score = result.get("score", 0.0)
        issue_count = len(result.get("issues", []))

        log_method = self._logger.info if passed else self._logger.warning
        log_method(
            "Validation result: event_type=%s, validator=%s, passed=%s, score=%.2f, issues=%d",
            event_type,
            validator_type,
            passed,
            score,
            issue_count,
        )


class CompositeValidatorHook(ValidatorHook):
    """Combines multiple validator hooks into a single hook.

    Useful when you need to send validation events to multiple destinations
    (e.g., logging, metrics, and alerting systems).

    Example:
        hook = CompositeValidatorHook([
            LoggingValidatorHook(),
            MetricsValidatorHook(),
            AlertingValidatorHook(),
        ])
    """

    def __init__(self, hooks: Optional[List[ValidatorHook]] = None) -> None:
        """Initialize the composite hook.

        Args:
            hooks: List of hooks to call. Can be added later with add_hook().
        """
        self._hooks: List[ValidatorHook] = hooks or []

    def add_hook(self, hook: ValidatorHook) -> None:
        """Add a hook to the composite.

        Args:
            hook: The hook to add.
        """
        if hook not in self._hooks:
            self._hooks.append(hook)

    def remove_hook(self, hook: ValidatorHook) -> None:
        """Remove a hook from the composite.

        Args:
            hook: The hook to remove.
        """
        if hook in self._hooks:
            self._hooks.remove(hook)

    def on_spawn(self, event: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Call on_spawn on all registered hooks.

        Args:
            event: The event data that triggered validator spawning.
            config: Configuration used for this validation run.
        """
        for hook in self._hooks:
            try:
                hook.on_spawn(event, config)
            except Exception:
                # Don't let one hook failure break the chain
                pass

    def on_result(self, event: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Call on_result on all registered hooks.

        Args:
            event: The original event that was validated.
            result: The validation result.
        """
        for hook in self._hooks:
            try:
                hook.on_result(event, result)
            except Exception:
                # Don't let one hook failure break the chain
                pass


class ValidatorHooksIntegration:
    """Integrates validators with the trace pipeline.

    This class provides the bridge between trace events and the validator system,
    automatically spawning validators for qualifying events and building
    framework-specific context for CrewAI and LangGraph events.

    Example:
        from arzule_ingest.validators import ValidatorClient, ValidationLevel

        client = ValidatorClient(
            endpoint="https://api.arzule.com/v1/validators",
            api_key="your-api-key",
            project_id="your-project-id",
        )

        integration = ValidatorHooksIntegration(
            client=client,
            auto_validate=True,
            validation_level=ValidationLevel.STANDARD,
        )

        # Called automatically when trace events are emitted
        integration.on_event(
            run=run,
            event_type="crewai.agent.execution.complete",
            event_data={"agent_role": "researcher", ...},
            span_id="span-123",
        )
    """

    def __init__(
        self,
        client: "ValidatorClient",
        auto_validate: bool = True,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
    ) -> None:
        """Initialize the validator hooks integration.

        Args:
            client: The ValidatorClient instance to use for spawning validators.
            auto_validate: If True, automatically spawn validators for qualifying events.
            validation_level: The default validation level to use.
        """
        self.client = client
        self.auto_validate = auto_validate
        self.validation_level = validation_level

    def on_event(
        self,
        run: Any,
        event_type: str,
        event_data: Dict[str, Any],
        span_id: str,
    ) -> None:
        """Called when a trace event is emitted. Spawns validators for qualifying events.

        Args:
            run: The ArzuleRun instance (must have a run_id attribute).
            event_type: The type of event (e.g., "crewai.agent.execution.complete").
            event_data: The event payload data.
            span_id: The span ID associated with this event.
        """
        if not self.auto_validate:
            return

        from .event_mapping import get_validators_for_event

        validators = get_validators_for_event(event_type)

        if not validators:
            return

        # Build context for framework-specific events
        context = self._build_framework_context(event_type, event_data)

        try:
            self.client.spawn_validators(
                session_id=run.run_id,
                agent_id=span_id,
                event_type=event_type,
                event_data=event_data,
                validators=validators,
                level=self.validation_level,
                context=context,
            )
        except Exception as e:
            print(f"[arzule] Failed to spawn validators: {e}", file=sys.stderr)

    def _build_framework_context(
        self,
        event_type: str,
        event_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Build framework-specific context for validators.

        This method extracts relevant context from event data based on the
        framework that generated the event. CrewAI events include agent roles,
        goals, and delegation information. LangGraph events include graph
        structure, node names, and state channel information.

        Args:
            event_type: The type of event.
            event_data: The event payload data.

        Returns:
            A dict of framework-specific context, or None if no context applies.
        """
        context: Dict[str, Any] = {}

        if event_type.startswith("crewai."):
            # CrewAI context
            context["agent_role"] = event_data.get("agent_role")
            context["agent_goal"] = event_data.get("agent_goal")
            context["agent_backstory"] = event_data.get("agent_backstory")
            context["allowed_tools"] = event_data.get("tools", [])
            context["task_description"] = event_data.get("task_description")

            # For delegation events
            if "delegation" in event_type:
                context["allowed_delegations"] = event_data.get("allowed_delegations", [])
                context["delegation_chain"] = event_data.get("delegation_chain", [])

        elif event_type.startswith("langgraph."):
            # LangGraph context
            context["graph_name"] = event_data.get("graph_name")
            context["node_name"] = event_data.get("node_name")
            context["state_channels"] = event_data.get("state_channels", {})
            context["state_type"] = event_data.get("state_type")

            # For edge events
            if "edge" in event_type:
                context["allowed_edges"] = event_data.get("allowed_edges", [])
                context["conditional_edges"] = event_data.get("conditional_edges", {})

            # For send/fan-out events
            if "send" in event_type or "fanout" in event_type:
                context["allowed_nodes"] = event_data.get("allowed_nodes", [])
                context["send_targets"] = event_data.get("send_targets", [])

        return context if context else None


def get_validator_hooks() -> Optional[ValidatorHooksIntegration]:
    """Get the configured validator hooks instance.

    Returns:
        The global ValidatorHooksIntegration instance, or None if not configured.

    Example:
        hooks = get_validator_hooks()
        if hooks:
            hooks.on_event(run, event_type, event_data, span_id)
    """
    return _hooks_instance


def configure_validator_hooks(
    client: "ValidatorClient",
    auto_validate: bool = True,
    validation_level: ValidationLevel = ValidationLevel.STANDARD,
) -> ValidatorHooksIntegration:
    """Configure the global validator hooks instance.

    This function creates and sets the global ValidatorHooksIntegration instance
    that can be retrieved later with get_validator_hooks().

    Args:
        client: The ValidatorClient instance to use for spawning validators.
        auto_validate: If True, automatically spawn validators for qualifying events.
        validation_level: The default validation level to use.

    Returns:
        The newly configured ValidatorHooksIntegration instance.

    Example:
        from arzule_ingest.validators import ValidatorClient, ValidationLevel

        client = ValidatorClient(
            endpoint="https://api.arzule.com/v1/validators",
            api_key="your-api-key",
            project_id="your-project-id",
        )

        hooks = configure_validator_hooks(
            client=client,
            auto_validate=True,
            validation_level=ValidationLevel.THOROUGH,
        )
    """
    global _hooks_instance
    _hooks_instance = ValidatorHooksIntegration(
        client=client,
        auto_validate=auto_validate,
        validation_level=validation_level,
    )
    return _hooks_instance
