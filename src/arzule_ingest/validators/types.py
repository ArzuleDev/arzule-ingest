"""Type definitions for the validators module.

This module defines the data structures used for communication with the
validators backend API and for passing data between hook handlers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Severity(str, Enum):
    """Issue severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ValidationDecision(str, Enum):
    """Aggregate validation decision from the backend."""

    CONTINUE = "continue"
    BLOCK = "block"
    REVIEW = "review"
    PARTIAL = "partial"


@dataclass
class ValidatorIssue:
    """A single issue identified by a validator.

    Attributes:
        severity: Issue severity level.
        category: Category of the issue (e.g., "injection", "scope_creep").
        description: Human-readable description of the issue.
        location: Optional location in the code/prompt where issue was found.
        recommendation: Suggested fix or mitigation.
        validator_type: Which validator identified this issue.
    """

    severity: Severity
    category: str
    description: str
    recommendation: str
    location: Optional[str] = None
    validator_type: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidatorIssue:
        """Create ValidatorIssue from API response dict."""
        severity_str = data.get("severity", "info").lower()
        try:
            severity = Severity(severity_str)
        except ValueError:
            severity = Severity.INFO

        return cls(
            severity=severity,
            category=data.get("category", "unknown"),
            description=data.get("description", ""),
            recommendation=data.get("recommendation", ""),
            location=data.get("location"),
            validator_type=data.get("validator_type"),
        )


@dataclass
class ValidatorResult:
    """Result from a single validator.

    Attributes:
        validator_id: Unique identifier for this validator run.
        validator_type: Type of validator (e.g., "security", "correctness").
        status: Current status (pending, running, completed, failed).
        score: Validation score (0.0-1.0), None if incomplete.
        passed: Whether this validator passed, None if incomplete.
        issues: List of issues identified by this validator.
        recommendations: List of recommendations.
    """

    validator_id: str
    validator_type: str
    status: str
    score: Optional[float] = None
    passed: Optional[bool] = None
    issues: list[ValidatorIssue] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidatorResult:
        """Create ValidatorResult from API response dict."""
        issues = []
        if data.get("issues"):
            issues = [ValidatorIssue.from_dict(i) for i in data["issues"]]

        return cls(
            validator_id=data.get("validator_id", ""),
            validator_type=data.get("validator_type", ""),
            status=data.get("status", "unknown"),
            score=data.get("score"),
            passed=data.get("passed"),
            issues=issues,
            recommendations=data.get("recommendations", []),
        )


@dataclass
class ValidationResult:
    """Aggregated validation result from all validators.

    This is the main result type returned by the get_results API.

    Attributes:
        aggregate_score: Weighted average score across all validators.
        passed: Whether all validators passed and no critical issues found.
        decision: Recommended action (continue, block, review, partial).
        validators_completed: Number of validators that finished.
        validators_total: Total number of validators spawned.
        critical_issues: List of critical severity issues.
        high_issues: List of high severity issues.
        recommendations: Deduplicated recommendations from all validators.
        validator_results: Individual results from each validator.
    """

    aggregate_score: float
    passed: bool
    decision: ValidationDecision
    validators_completed: int
    validators_total: int
    critical_issues: list[ValidatorIssue] = field(default_factory=list)
    high_issues: list[ValidatorIssue] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    validator_results: list[ValidatorResult] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationResult:
        """Create ValidationResult from API response dict."""
        decision_str = data.get("decision", "continue").lower()
        try:
            decision = ValidationDecision(decision_str)
        except ValueError:
            decision = ValidationDecision.CONTINUE

        critical_issues = [
            ValidatorIssue.from_dict(i) for i in data.get("critical_issues", [])
        ]
        high_issues = [
            ValidatorIssue.from_dict(i) for i in data.get("high_issues", [])
        ]
        validator_results = [
            ValidatorResult.from_dict(r) for r in data.get("validator_results", [])
        ]

        return cls(
            aggregate_score=data.get("aggregate_score", 0.0),
            passed=data.get("passed", True),
            decision=decision,
            validators_completed=data.get("validators_completed", 0),
            validators_total=data.get("validators_total", 0),
            critical_issues=critical_issues,
            high_issues=high_issues,
            recommendations=data.get("recommendations", []),
            validator_results=validator_results,
        )

    def should_block(self) -> bool:
        """Check if the result indicates blocking is recommended."""
        return self.decision == ValidationDecision.BLOCK or len(self.critical_issues) > 0

    def get_summary(self) -> str:
        """Get a human-readable summary of the validation result."""
        lines = [
            f"Validation: {self.decision.value.upper()}",
            f"Score: {self.aggregate_score:.2f}",
            f"Validators: {self.validators_completed}/{self.validators_total} complete",
        ]

        if self.critical_issues:
            lines.append(f"Critical Issues: {len(self.critical_issues)}")
            for issue in self.critical_issues[:3]:
                lines.append(f"  - [{issue.validator_type}] {issue.description[:100]}")

        if self.high_issues:
            lines.append(f"High Issues: {len(self.high_issues)}")

        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations[:3]:
                lines.append(f"  - {rec[:100]}")

        return "\n".join(lines)


@dataclass
class SpawnRequest:
    """Request payload for spawning validators.

    Attributes:
        project_id: Arzule project ID.
        session_id: Claude Code session ID.
        agent_id: Subagent ID (from SubagentStart).
        subagent_type: Type of subagent (e.g., "Explore", "general-purpose").
        description: Task description from Task tool.
        prompt: Full prompt text from Task tool.
        transcript_path: Path to subagent's transcript file.
        validators: List of validator types to run.
        validation_level: Intensity level (minimal, standard, thorough).
        model: Claude model to use for validation.
        metadata: Additional metadata for tracking.
    """

    project_id: str
    session_id: str
    agent_id: str
    subagent_type: str
    description: str
    prompt: str
    transcript_path: str
    validators: list[str]
    validation_level: str = "standard"
    model: str = "claude-3-5-haiku-20241022"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API request."""
        return {
            "project_id": self.project_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "subagent_type": self.subagent_type,
            "description": self.description,
            "prompt": self.prompt,
            "transcript_path": self.transcript_path,
            "validators": self.validators,
            "validation_level": self.validation_level,
            "model": self.model,
            "metadata": self.metadata,
        }


@dataclass
class SpawnResponse:
    """Response from spawning validators.

    Attributes:
        status: Status of the spawn request (e.g., "spawned").
        agent_id: The agent ID that validators were spawned for.
        validator_ids: List of validator run IDs.
        validators: List of validator types that were spawned.
    """

    status: str
    agent_id: str
    validator_ids: list[str]
    validators: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpawnResponse:
        """Create SpawnResponse from API response dict."""
        return cls(
            status=data.get("status", "unknown"),
            agent_id=data.get("agent_id", ""),
            validator_ids=data.get("validator_ids", []),
            validators=data.get("validators", []),
        )


@dataclass
class HookContext:
    """Context data passed between hook invocations.

    This class tracks state across PreToolUse and SubagentStop hooks
    for proper validator correlation.

    Attributes:
        session_id: Claude Code session ID.
        tool_use_id: Tool use ID from PreToolUse.
        agent_id: Subagent ID from SubagentStart/SubagentStop.
        subagent_type: Type of subagent.
        spawn_response: Response from spawn_validators, if spawned.
        validation_result: Result from get_results, if retrieved.
    """

    session_id: str
    tool_use_id: Optional[str] = None
    agent_id: Optional[str] = None
    subagent_type: Optional[str] = None
    spawn_response: Optional[SpawnResponse] = None
    validation_result: Optional[ValidationResult] = None
