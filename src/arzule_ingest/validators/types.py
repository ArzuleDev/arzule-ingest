"""Type definitions for the validators module."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ValidationLevel(Enum):
    """Level of validation thoroughness."""

    MINIMAL = "minimal"
    STANDARD = "standard"
    THOROUGH = "thorough"


class ValidatorType(Enum):
    """Types of validators available."""

    # Core validators
    SECURITY = "security"
    CORRECTNESS = "correctness"
    SCOPE = "scope"
    COMPLIANCE = "compliance"
    GROUNDEDNESS = "groundedness"

    # CrewAI-specific validators
    CREWAI_AGENT = "crewai_agent"
    CREWAI_DELEGATION = "crewai_delegation"
    CREWAI_TOOL_USAGE = "crewai_tool_usage"
    CREWAI_FLOW = "crewai_flow"

    # LangGraph-specific validators
    LANGGRAPH_NODE = "langgraph_node"
    LANGGRAPH_EDGE = "langgraph_edge"
    LANGGRAPH_STATE = "langgraph_state"
    LANGGRAPH_SEND = "langgraph_send"


@dataclass
class ValidationIssue:
    """A single validation issue detected by a validator.

    Attributes:
        code: Unique identifier for the issue type (e.g., "SEC001", "CORR002").
        severity: Issue severity level - "critical", "high", "medium", or "low".
        message: Human-readable description of the issue.
        location: Optional location context (e.g., file path, line number, field name).
        suggestion: Optional remediation suggestion.
    """

    code: str
    severity: str  # critical, high, medium, low
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate severity value."""
        valid_severities = {"critical", "high", "medium", "low"}
        if self.severity not in valid_severities:
            raise ValueError(
                f"Invalid severity '{self.severity}'. Must be one of: {valid_severities}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }
        if self.location is not None:
            result["location"] = self.location
        if self.suggestion is not None:
            result["suggestion"] = self.suggestion
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationIssue":
        """Create a ValidationIssue from a dictionary."""
        return cls(
            code=data["code"],
            severity=data["severity"],
            message=data["message"],
            location=data.get("location"),
            suggestion=data.get("suggestion"),
        )


@dataclass
class ValidationResult:
    """Result from running a validator.

    Attributes:
        validator_type: The type of validator that produced this result.
        score: Validation score from 0.0 (failed) to 1.0 (passed).
        passed: Whether validation passed based on threshold.
        issues: List of validation issues found.
        recommendations: List of improvement recommendations.
        metadata: Additional validator-specific metadata.
    """

    validator_type: str
    score: float  # 0.0 - 1.0
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate score range."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")

    @property
    def critical_issues(self) -> List[ValidationIssue]:
        """Get only critical severity issues."""
        return [issue for issue in self.issues if issue.severity == "critical"]

    @property
    def high_issues(self) -> List[ValidationIssue]:
        """Get only high severity issues."""
        return [issue for issue in self.issues if issue.severity == "high"]

    @property
    def issue_count_by_severity(self) -> Dict[str, int]:
        """Get count of issues grouped by severity."""
        counts: Dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for issue in self.issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "validator_type": self.validator_type,
            "score": self.score,
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResult":
        """Create a ValidationResult from a dictionary."""
        issues = [
            ValidationIssue.from_dict(issue_data) for issue_data in data.get("issues", [])
        ]
        return cls(
            validator_type=data["validator_type"],
            score=data["score"],
            passed=data["passed"],
            issues=issues,
            recommendations=data.get("recommendations", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AggregatedValidationResult:
    """Aggregated results from multiple validators.

    Attributes:
        session_id: The session ID these results belong to.
        agent_id: The agent ID these results belong to.
        results: Individual validation results by validator type.
        overall_score: Weighted average score across all validators.
        overall_passed: Whether all validators passed.
        timestamp: ISO timestamp when results were aggregated.
    """

    session_id: str
    agent_id: str
    results: Dict[str, ValidationResult] = field(default_factory=dict)
    overall_score: float = 0.0
    overall_passed: bool = True
    timestamp: Optional[str] = None

    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result and recalculate aggregates."""
        self.results[result.validator_type] = result
        self._recalculate_aggregates()

    def _recalculate_aggregates(self) -> None:
        """Recalculate overall score and passed status."""
        if not self.results:
            self.overall_score = 0.0
            self.overall_passed = True
            return

        total_score = sum(r.score for r in self.results.values())
        self.overall_score = total_score / len(self.results)
        self.overall_passed = all(r.passed for r in self.results.values())

    @property
    def all_issues(self) -> List[ValidationIssue]:
        """Get all issues from all validators."""
        issues: List[ValidationIssue] = []
        for result in self.results.values():
            issues.extend(result.issues)
        return issues

    @property
    def all_recommendations(self) -> List[str]:
        """Get all recommendations from all validators."""
        recommendations: List[str] = []
        for result in self.results.values():
            recommendations.extend(result.recommendations)
        return recommendations

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "overall_score": self.overall_score,
            "overall_passed": self.overall_passed,
            "timestamp": self.timestamp,
        }
