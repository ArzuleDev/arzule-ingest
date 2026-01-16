"""HTTP client for interacting with Arzule Validator API."""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from .types import ValidationLevel, ValidationResult, ValidatorType


class ValidatorClientError(Exception):
    """Base exception for validator client errors."""

    pass


class ValidatorTimeoutError(ValidatorClientError):
    """Raised when a validator operation times out."""

    pass


class ValidatorAuthError(ValidatorClientError):
    """Raised when authentication fails."""

    pass


class ValidatorClient:
    """Client for interacting with Arzule Validator API.

    This client handles communication with the validator service, including
    spawning validators for events and retrieving validation results.

    Example:
        client = ValidatorClient(
            endpoint="https://api.arzule.com/v1/validators",
            api_key="your-api-key",
            project_id="your-project-id",
        )

        # Spawn validators for an event
        spawn_response = client.spawn_validators(
            session_id="session-123",
            agent_id="agent-456",
            event_type="tool.call.end",
            event_data={"tool_name": "search", "result": "..."},
            validators=["security", "correctness"],
        )

        # Later, retrieve results
        results = client.get_results(
            session_id="session-123",
            agent_id="agent-456",
            wait_timeout_ms=5000,
        )
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        project_id: str,
        timeout_ms: int = 5000,
    ) -> None:
        """Initialize the validator client.

        Args:
            endpoint: Base URL for the validator API.
            api_key: API key for authentication.
            project_id: Project ID for scoping requests.
            timeout_ms: Default timeout for HTTP requests in milliseconds.
        """
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.project_id = project_id
        self.timeout = timeout_ms / 1000  # Convert to seconds for httpx

        # Lazy import httpx to avoid import-time dependencies
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Get or create the HTTP client (lazy initialization)."""
        if self._client is None:
            try:
                import httpx

                self._client = httpx.Client(
                    timeout=self.timeout,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "X-Arzule-Project-Id": self.project_id,
                    },
                )
            except ImportError:
                raise ValidatorClientError(
                    "httpx is required for ValidatorClient. "
                    "Install it with: pip install httpx"
                )
        return self._client

    def _build_url(self, path: str) -> str:
        """Build full URL from path.

        Args:
            path: API path (e.g., "/spawn", "/results").

        Returns:
            Full URL string.
        """
        return urljoin(self.endpoint + "/", path.lstrip("/"))

    def spawn_validators(
        self,
        session_id: str,
        agent_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        validators: List[str],
        level: ValidationLevel = ValidationLevel.STANDARD,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Spawn validators for an event (non-blocking).

        This method sends an event to the validator service and returns
        immediately. The validators run asynchronously and results can
        be retrieved later using get_results().

        Args:
            session_id: The session ID this event belongs to.
            agent_id: The agent ID that generated this event.
            event_type: Type of event (e.g., "tool.call.end").
            event_data: The event payload to validate.
            validators: List of validator types to run.
            level: Validation thoroughness level.
            context: Optional additional context for validators.

        Returns:
            Response dict containing:
                - request_id: Unique ID for tracking this validation request
                - validators_spawned: List of validators that were spawned
                - estimated_completion_ms: Estimated time until results ready

        Raises:
            ValidatorClientError: If the request fails.
            ValidatorAuthError: If authentication fails.
        """
        client = self._get_client()

        # Validate validator types
        valid_types = {v.value for v in ValidatorType}
        invalid_validators = [v for v in validators if v not in valid_types]
        if invalid_validators:
            print(
                f"[arzule] Warning: Unknown validator types: {invalid_validators}",
                file=sys.stderr,
            )

        payload = {
            "session_id": session_id,
            "agent_id": agent_id,
            "event_type": event_type,
            "event_data": event_data,
            "validators": validators,
            "level": level.value if isinstance(level, ValidationLevel) else level,
            "project_id": self.project_id,
        }

        if context:
            payload["context"] = context

        try:
            response = client.post(self._build_url("/spawn"), json=payload)

            if response.status_code == 401:
                raise ValidatorAuthError("Invalid API key or unauthorized")
            if response.status_code == 403:
                raise ValidatorAuthError("Access forbidden for this project")

            response.raise_for_status()
            return response.json()

        except ValidatorAuthError:
            raise
        except Exception as e:
            error_type = type(e).__name__
            raise ValidatorClientError(f"Failed to spawn validators: {error_type}: {e}")

    def get_results(
        self,
        session_id: str,
        agent_id: str,
        wait_timeout_ms: int = 5000,
        request_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get validation results (with optional wait).

        This method retrieves validation results for a session/agent pair.
        If results are not yet ready, it can optionally wait for them.

        Args:
            session_id: The session ID to get results for.
            agent_id: The agent ID to get results for.
            wait_timeout_ms: How long to wait for results (0 = don't wait).
            request_id: Optional specific request ID to get results for.

        Returns:
            Dict containing validation results, or None if not ready and
            wait_timeout_ms was 0. Result dict contains:
                - session_id: The session ID
                - agent_id: The agent ID
                - results: Dict mapping validator_type to ValidationResult dict
                - overall_score: Aggregated score across all validators
                - overall_passed: Boolean if all validators passed
                - timestamp: ISO timestamp of result aggregation

        Raises:
            ValidatorClientError: If the request fails.
            ValidatorTimeoutError: If results not ready within timeout.
        """
        client = self._get_client()

        params: Dict[str, Any] = {
            "session_id": session_id,
            "agent_id": agent_id,
            "project_id": self.project_id,
        }

        if request_id:
            params["request_id"] = request_id

        # If wait is requested, use long-polling
        if wait_timeout_ms > 0:
            params["wait_ms"] = wait_timeout_ms

        try:
            # Adjust client timeout for wait
            timeout = (wait_timeout_ms / 1000) + self.timeout if wait_timeout_ms > 0 else self.timeout

            response = client.get(
                self._build_url("/results"),
                params=params,
                timeout=timeout,
            )

            if response.status_code == 401:
                raise ValidatorAuthError("Invalid API key or unauthorized")
            if response.status_code == 404:
                # No results found
                return None
            if response.status_code == 408:
                # Request timed out waiting for results
                raise ValidatorTimeoutError(
                    f"Results not ready within {wait_timeout_ms}ms"
                )

            response.raise_for_status()
            return response.json()

        except (ValidatorAuthError, ValidatorTimeoutError):
            raise
        except Exception as e:
            error_type = type(e).__name__
            raise ValidatorClientError(f"Failed to get results: {error_type}: {e}")

    def get_result_for_validator(
        self,
        session_id: str,
        agent_id: str,
        validator_type: str,
        wait_timeout_ms: int = 5000,
    ) -> Optional[ValidationResult]:
        """Get result for a specific validator type.

        Convenience method to get and parse a single validator's result.

        Args:
            session_id: The session ID to get results for.
            agent_id: The agent ID to get results for.
            validator_type: The specific validator type to get.
            wait_timeout_ms: How long to wait for results.

        Returns:
            ValidationResult for the specified validator, or None if not found.
        """
        results = self.get_results(
            session_id=session_id,
            agent_id=agent_id,
            wait_timeout_ms=wait_timeout_ms,
        )

        if not results:
            return None

        validator_results = results.get("results", {})
        if validator_type not in validator_results:
            return None

        return ValidationResult.from_dict(validator_results[validator_type])

    def poll_results(
        self,
        session_id: str,
        agent_id: str,
        poll_interval_ms: int = 500,
        max_wait_ms: int = 30000,
    ) -> Optional[Dict[str, Any]]:
        """Poll for validation results until ready or timeout.

        Alternative to long-polling get_results() for clients that prefer
        active polling.

        Args:
            session_id: The session ID to get results for.
            agent_id: The agent ID to get results for.
            poll_interval_ms: Interval between poll attempts.
            max_wait_ms: Maximum time to wait before giving up.

        Returns:
            Validation results dict, or None if not ready within max_wait_ms.
        """
        start_time = time.monotonic()
        max_wait_seconds = max_wait_ms / 1000
        poll_interval_seconds = poll_interval_ms / 1000

        while (time.monotonic() - start_time) < max_wait_seconds:
            try:
                results = self.get_results(
                    session_id=session_id,
                    agent_id=agent_id,
                    wait_timeout_ms=0,  # Don't wait, just check
                )
                if results is not None:
                    return results
            except ValidatorClientError:
                # Ignore transient errors during polling
                pass

            time.sleep(poll_interval_seconds)

        return None

    def cancel_validators(
        self,
        session_id: str,
        agent_id: str,
        request_id: Optional[str] = None,
    ) -> bool:
        """Cancel pending validators.

        Args:
            session_id: The session ID to cancel validators for.
            agent_id: The agent ID to cancel validators for.
            request_id: Optional specific request to cancel.

        Returns:
            True if cancellation was successful.

        Raises:
            ValidatorClientError: If the request fails.
        """
        client = self._get_client()

        payload = {
            "session_id": session_id,
            "agent_id": agent_id,
            "project_id": self.project_id,
        }

        if request_id:
            payload["request_id"] = request_id

        try:
            response = client.post(self._build_url("/cancel"), json=payload)
            response.raise_for_status()
            return True
        except Exception as e:
            error_type = type(e).__name__
            raise ValidatorClientError(f"Failed to cancel validators: {error_type}: {e}")

    def health_check(self) -> Dict[str, Any]:
        """Check if the validator service is healthy.

        Returns:
            Dict with health status information.

        Raises:
            ValidatorClientError: If the health check fails.
        """
        client = self._get_client()

        try:
            response = client.get(self._build_url("/health"))
            response.raise_for_status()
            return response.json()
        except Exception as e:
            error_type = type(e).__name__
            raise ValidatorClientError(f"Health check failed: {error_type}: {e}")

    def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "ValidatorClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
