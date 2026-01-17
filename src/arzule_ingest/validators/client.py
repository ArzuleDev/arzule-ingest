"""API client for the validators backend service.

This module provides the ValidatorsClient class for communication with
the Arzule validators API endpoints:

- POST /v1/validators/spawn - Spawn validator agents for a subagent
- GET /v1/validators/results/{session_id}/{agent_id} - Get validation results

The client handles authentication, request serialization, response parsing,
and error handling.

Example:
    >>> from arzule_ingest.validators import ValidatorsClient, SpawnRequest
    >>>
    >>> client = ValidatorsClient()
    >>> request = SpawnRequest(
    ...     project_id="proj_123",
    ...     session_id="sess_abc",
    ...     agent_id="agent_xyz",
    ...     subagent_type="general-purpose",
    ...     description="Implement feature",
    ...     prompt="Add authentication to the API",
    ...     transcript_path="/path/to/transcript.jsonl",
    ...     validators=["security", "correctness"],
    ... )
    >>> response = client.spawn_validators(request)
    >>> print(f"Spawned {len(response.validator_ids)} validators")
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .types import (
    SpawnRequest,
    SpawnResponse,
    ValidationResult,
)

# Default API endpoint
_DEFAULT_VALIDATORS_URL = "https://validators.arzule.com"

# Timeouts (in seconds) - increased for Lambda cold starts (~12-15s)
DEFAULT_SPAWN_TIMEOUT = 20.0
DEFAULT_RESULTS_TIMEOUT = 25.0
DEFAULT_WAIT_TIMEOUT_MS = 20000
MAX_WAIT_TIMEOUT_MS = 60000


class ValidatorsClientError(Exception):
    """Base exception for ValidatorsClient errors."""

    pass


class AuthenticationError(ValidatorsClientError):
    """Raised when API authentication fails."""

    pass


class ValidationError(ValidatorsClientError):
    """Raised when request validation fails."""

    pass


class NetworkError(ValidatorsClientError):
    """Raised when network communication fails."""

    pass


class ValidatorsClient:
    """HTTP client for the Arzule validators API.

    This client provides methods to spawn validators and retrieve results.
    It reads configuration from environment variables:

    - ARZULE_VALIDATORS_URL: Base URL for validators API
    - ARZULE_API_KEY: API key for authentication
    - ARZULE_PROJECT_ID: Project ID (used if not provided in requests)

    The client uses urllib for HTTP requests to avoid external dependencies.

    Attributes:
        base_url: Base URL for the validators API.
        api_key: API key for authentication.
        project_id: Default project ID for requests.
        debug: Enable debug logging.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        """Initialize the validators client.

        Args:
            base_url: Override for validators API URL.
            api_key: Override for API key.
            project_id: Override for project ID.
            debug: Enable debug logging to stderr.
        """
        self.base_url = (
            base_url
            or os.environ.get("ARZULE_VALIDATORS_URL")
            or _DEFAULT_VALIDATORS_URL
        ).rstrip("/")

        self.api_key = api_key or os.environ.get("ARZULE_API_KEY")
        self.project_id = project_id or os.environ.get("ARZULE_PROJECT_ID")
        self.debug = debug

    def _log(self, message: str) -> None:
        """Log debug message if debug mode is enabled."""
        if self.debug:
            print(f"[validators-client] {message}", file=sys.stderr)

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def _make_request(
        self,
        method: str,
        path: str,
        data: Optional[dict[str, Any]] = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        """Make an HTTP request to the validators API.

        Args:
            method: HTTP method (GET, POST).
            path: API path (e.g., "/v1/validators/spawn").
            data: Request body data (for POST).
            timeout: Request timeout in seconds.

        Returns:
            Parsed JSON response.

        Raises:
            AuthenticationError: If authentication fails (401).
            ValidationError: If request validation fails (400).
            NetworkError: If network communication fails.
            ValidatorsClientError: For other API errors.
        """
        url = f"{self.base_url}{path}"
        headers = self._get_headers()

        body = None
        if data is not None:
            body = json.dumps(data).encode("utf-8")

        self._log(f"{method} {url}")

        request = Request(url, data=body, headers=headers, method=method)

        try:
            with urlopen(request, timeout=timeout) as response:
                response_data = response.read().decode("utf-8")
                return json.loads(response_data) if response_data else {}

        except HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass

            self._log(f"HTTP error {e.code}: {error_body}")

            if e.code == 401:
                raise AuthenticationError(
                    f"Authentication failed: {error_body or 'Invalid API key'}"
                )
            elif e.code == 400:
                try:
                    error_data = json.loads(error_body)
                    message = error_data.get("message", error_body)
                except json.JSONDecodeError:
                    message = error_body
                raise ValidationError(f"Validation failed: {message}")
            elif e.code == 404:
                raise ValidatorsClientError(f"Not found: {path}")
            else:
                raise ValidatorsClientError(
                    f"API error {e.code}: {error_body or e.reason}"
                )

        except URLError as e:
            self._log(f"Network error: {e.reason}")
            raise NetworkError(f"Network error: {e.reason}")

        except Exception as e:
            self._log(f"Unexpected error: {e}")
            raise ValidatorsClientError(f"Unexpected error: {e}")

    def spawn_validators(
        self,
        request: SpawnRequest,
        timeout: float = DEFAULT_SPAWN_TIMEOUT,
    ) -> SpawnResponse:
        """Spawn validator agents for a subagent.

        This method sends a request to the validators API to spawn
        validation agents that will analyze the subagent's task and prompt.
        The validators run asynchronously - this method returns immediately
        with validator IDs.

        Args:
            request: SpawnRequest with subagent details and validator config.
            timeout: Request timeout in seconds.

        Returns:
            SpawnResponse with validator IDs.

        Raises:
            ValidationError: If request validation fails.
            AuthenticationError: If API key is invalid.
            NetworkError: If network communication fails.

        Example:
            >>> request = SpawnRequest(
            ...     project_id="proj_123",
            ...     session_id="sess_abc",
            ...     agent_id="agent_xyz",
            ...     subagent_type="general-purpose",
            ...     description="Update database",
            ...     prompt="Add new column to users table",
            ...     transcript_path="/path/to/transcript.jsonl",
            ...     validators=["security", "correctness"],
            ... )
            >>> response = client.spawn_validators(request)
            >>> print(f"Validator IDs: {response.validator_ids}")
        """
        self._log(
            f"Spawning validators for agent {request.agent_id}: {request.validators}"
        )

        response_data = self._make_request(
            method="POST",
            path="/v1/validators/spawn",
            data=request.to_dict(),
            timeout=timeout,
        )

        return SpawnResponse.from_dict(response_data)

    def get_results(
        self,
        session_id: str,
        agent_id: str,
        wait_timeout_ms: int = DEFAULT_WAIT_TIMEOUT_MS,
        request_timeout: float = DEFAULT_RESULTS_TIMEOUT,
    ) -> Optional[ValidationResult]:
        """Get validation results for a subagent.

        This method polls the validators API for results. The API
        server handles the polling internally - it waits up to
        wait_timeout_ms for validators to complete before returning.

        Args:
            session_id: Claude Code session ID.
            agent_id: Subagent ID to get results for.
            wait_timeout_ms: How long the API should wait for completion.
            request_timeout: HTTP request timeout in seconds.

        Returns:
            ValidationResult if validators exist, None if not found.

        Raises:
            AuthenticationError: If API key is invalid.
            NetworkError: If network communication fails.

        Example:
            >>> result = client.get_results(
            ...     session_id="sess_abc",
            ...     agent_id="agent_xyz",
            ...     wait_timeout_ms=5000,
            ... )
            >>> if result:
            ...     print(f"Score: {result.aggregate_score}")
            ...     print(f"Decision: {result.decision}")
        """
        self._log(f"Getting results for agent {agent_id} in session {session_id}")

        # Clamp wait timeout
        wait_ms = min(wait_timeout_ms, MAX_WAIT_TIMEOUT_MS)

        # Build query parameters
        query = f"?wait_timeout_ms={wait_ms}"
        if self.project_id:
            query += f"&project_id={self.project_id}"

        path = f"/v1/validators/results/{session_id}/{agent_id}{query}"

        try:
            response_data = self._make_request(
                method="GET",
                path=path,
                timeout=request_timeout,
            )
            return ValidationResult.from_dict(response_data)

        except ValidatorsClientError as e:
            if "not found" in str(e).lower() or "404" in str(e):
                return None
            raise

    def get_results_with_polling(
        self,
        session_id: str,
        agent_id: str,
        max_wait_seconds: float = 10.0,
        poll_interval_seconds: float = 1.0,
    ) -> Optional[ValidationResult]:
        """Get validation results with client-side polling.

        Unlike get_results which relies on server-side polling, this method
        polls the API repeatedly until all validators complete or timeout.

        This is useful when you need more control over the polling behavior
        or when the server-side polling doesn't meet your requirements.

        Args:
            session_id: Claude Code session ID.
            agent_id: Subagent ID to get results for.
            max_wait_seconds: Maximum time to wait for completion.
            poll_interval_seconds: Time between poll requests.

        Returns:
            ValidationResult if validators exist, None if not found.

        Example:
            >>> # Poll for up to 30 seconds, checking every 2 seconds
            >>> result = client.get_results_with_polling(
            ...     session_id="sess_abc",
            ...     agent_id="agent_xyz",
            ...     max_wait_seconds=30.0,
            ...     poll_interval_seconds=2.0,
            ... )
        """
        self._log(
            f"Polling results for agent {agent_id} "
            f"(max {max_wait_seconds}s, interval {poll_interval_seconds}s)"
        )

        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed >= max_wait_seconds:
                self._log(f"Polling timeout after {elapsed:.1f}s")
                break

            # Use minimal server-side wait since we're doing client-side polling
            result = self.get_results(
                session_id=session_id,
                agent_id=agent_id,
                wait_timeout_ms=500,  # Short server wait
            )

            if result is None:
                self._log("No validators found")
                return None

            # Check if all validators have completed
            if result.validators_completed >= result.validators_total:
                self._log(
                    f"All {result.validators_total} validators completed "
                    f"after {elapsed:.1f}s"
                )
                return result

            self._log(
                f"Progress: {result.validators_completed}/{result.validators_total} "
                f"complete ({elapsed:.1f}s elapsed)"
            )

            # Wait before next poll
            time.sleep(poll_interval_seconds)

        # Return partial results on timeout
        return self.get_results(
            session_id=session_id,
            agent_id=agent_id,
            wait_timeout_ms=0,
        )

    def health_check(self, timeout: float = 2.0) -> bool:
        """Check if the validators API is healthy.

        Args:
            timeout: Request timeout in seconds.

        Returns:
            True if API is healthy, False otherwise.
        """
        try:
            # Try to hit a lightweight endpoint
            self._make_request(
                method="GET",
                path="/health",
                timeout=timeout,
            )
            return True
        except Exception as e:
            self._log(f"Health check failed: {e}")
            return False
