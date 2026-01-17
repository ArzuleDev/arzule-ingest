"""
Tier 0: Deterministic Risk Gate

Decides whether a subagent call is worth validating, and at what intensity level.
This gate runs in < 1ms and provides a fast path to skip validation for low-risk
operations while ensuring high-risk operations receive appropriate scrutiny.

Risk Scoring Factors:
    1. Subagent Type (0-30): general-purpose=30, Explore=5, claude-code-guide=0
    2. Content Signals (0-40): Critical patterns (secrets, SQL, shell)=+15, High (database, production)=+8
    3. Prompt Complexity (0-20): Length > 2000=+10, multiple files=+5
    4. Historical Context (0-10): Previous failures=+5

Decision Thresholds:
    - SKIP (score 0-15): No validation
    - LITE (score 16-35): correctness only, Haiku model
    - STANDARD (score 36-60): correctness, scope, maybe security, Haiku model
    - DEEP (score 61-100): All 5 validators, Opus for security

Example:
    >>> from arzule_ingest.validators import should_validate, ValidationLevel
    >>> should_val, assessment = should_validate(
    ...     subagent_type="general-purpose",
    ...     description="Update database schema",
    ...     prompt="Modify the users table to add email column",
    ... )
    >>> if should_val:
    ...     print(f"Validate at level: {assessment.level.value}")
    ...     print(f"Validators: {assessment.recommended_validators}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ValidationLevel(Enum):
    """Validation intensity levels based on risk assessment."""

    SKIP = "skip"
    LITE = "lite"
    STANDARD = "standard"
    DEEP = "deep"


@dataclass
class RiskAssessment:
    """Result of risk assessment for a subagent call.

    Attributes:
        total_score: Aggregate risk score (0-100).
        level: Recommended validation level based on score.
        factors: Breakdown of score by risk factor.
        triggered_signals: List of specific risk signals detected.
        recommended_validators: List of validator types to run.
        recommended_model: Claude model ID to use for validation.
    """

    total_score: int
    level: ValidationLevel
    factors: Dict[str, int]
    triggered_signals: List[str]
    recommended_validators: List[str]
    recommended_model: str


class DeterministicRiskGate:
    """Fast (<1ms) risk assessment for subagent calls.

    This class provides a deterministic, rule-based assessment of risk for
    subagent calls. It does not use any LLM calls, ensuring consistent
    sub-millisecond performance.

    The risk score is computed from four factors:
        1. Subagent type: Known risk levels for different agent types
        2. Content signals: Regex patterns detecting sensitive operations
        3. Prompt complexity: Length and structure of the prompt
        4. Historical context: Session-level risk factors

    Example:
        >>> gate = DeterministicRiskGate()
        >>> assessment = gate.assess(
        ...     subagent_type="security-auditor",
        ...     description="Review authentication code",
        ...     prompt="Analyze the OAuth implementation for vulnerabilities",
        ... )
        >>> print(f"Risk score: {assessment.total_score}")
        >>> print(f"Level: {assessment.level.value}")
    """

    # Risk scores by subagent type (0-30 points)
    TYPE_RISK_SCORES: Dict[str, int] = {
        # High risk: general purpose and security-related
        "general-purpose": 30,
        "security-auditor": 25,
        "debugger": 25,
        # Medium-high risk: developers with code modification capabilities
        "backend-developer": 22,
        "fullstack-developer": 22,
        "devops-engineer": 20,
        "database-administrator": 18,
        # Medium risk: review and testing roles
        "code-reviewer": 15,
        "test-automator": 15,
        "refactoring-specialist": 15,
        # Low-medium risk: documentation and architecture
        "documentation-engineer": 10,
        "architect-reviewer": 8,
        # Low risk: exploration and research
        "Explore": 5,
        "Plan": 5,
        "research-analyst": 3,
        # Minimal risk: guidance only
        "claude-code-guide": 0,
    }

    # Critical patterns: highest risk operations (+15 points each)
    CRITICAL_PATTERNS: List[Tuple[str, str, int]] = [
        (r"/auth/|/security/|/api/|/admin/", "security_path", 15),
        (r"password|secret|token|credential|api[_-]?key", "secret_keyword", 15),
        (r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|TRUNCATE)\b", "sql_keyword", 15),
        (r"\brm\s+-rf|\bchmod\s+777|\bsudo\s+", "dangerous_command", 15),
        (r"\.env|\.pem|\.key|credentials", "sensitive_file", 15),
        (r"curl.*\|\s*(ba)?sh|wget.*\|\s*(ba)?sh", "remote_exec", 15),
    ]

    # High-risk patterns: significant risk operations (+8 points each)
    HIGH_PATTERNS: List[Tuple[str, str, int]] = [
        (r"database|migration|schema", "database_ops", 8),
        (r"deploy|production|prod\b", "production_ref", 8),
        (r"user\s*input|sanitize|validate|escape", "input_handling", 8),
        (r"encrypt|decrypt|hash|sign", "crypto_ops", 8),
        (r"payment|billing|stripe|financial", "financial_ops", 8),
    ]

    # Medium-risk patterns: moderate risk operations (+4 points each)
    MEDIUM_PATTERNS: List[Tuple[str, str, int]] = [
        (r"refactor|rewrite|restructure", "refactor_ops", 4),
        (r"\bfix\b|\bpatch\b|\bupdate\b", "modification_ops", 4),
        (r"create|generate|implement|build", "creation_ops", 4),
        (r"delete|remove|clean", "deletion_ops", 4),
    ]

    # Score thresholds for validation levels
    THRESHOLDS: Dict[str, int] = {
        "skip_max": 15,
        "lite_max": 35,
        "standard_max": 60,
    }

    # Model recommendations by validation level
    MODEL_HAIKU = "claude-3-5-haiku-20241022"
    MODEL_SONNET = "claude-sonnet-4-20250514"
    MODEL_OPUS = "claude-opus-4-5-20251101"

    def __init__(self, config: Optional[Dict[str, object]] = None) -> None:
        """Initialize the risk gate with optional configuration.

        Args:
            config: Optional configuration dictionary. Supported keys:
                - strict_mode (bool): If True, adds +20 to all risk scores.
                - custom_thresholds (dict): Override default thresholds.
                - custom_type_scores (dict): Override default type risk scores.
        """
        self.config = config or {}
        self.strict_mode: bool = bool(self.config.get("strict_mode", False))

        # Allow custom threshold overrides
        custom_thresholds = self.config.get("custom_thresholds")
        if isinstance(custom_thresholds, dict):
            self.thresholds = {**self.THRESHOLDS, **custom_thresholds}
        else:
            self.thresholds = self.THRESHOLDS.copy()

        # Allow custom type score overrides
        custom_type_scores = self.config.get("custom_type_scores")
        if isinstance(custom_type_scores, dict):
            self.type_scores = {**self.TYPE_RISK_SCORES, **custom_type_scores}
        else:
            self.type_scores = self.TYPE_RISK_SCORES.copy()

    def assess(
        self,
        subagent_type: str,
        description: str,
        prompt: str,
        session_context: Optional[Dict[str, object]] = None,
    ) -> RiskAssessment:
        """Assess risk for a subagent call.

        This method computes a risk score based on multiple factors and
        returns a recommendation for validation level and validators to use.

        Args:
            subagent_type: The type/name of the subagent being invoked.
            description: Short description of the task.
            prompt: Full prompt text being sent to the subagent.
            session_context: Optional session-level context with keys:
                - previous_failure (bool): Whether previous subagent failed.
                - subagent_count (int): Number of subagents invoked this session.
                - recent_validation_failures (int): Recent validation failures.

        Returns:
            RiskAssessment with score, level, and recommendations.

        Example:
            >>> gate = DeterministicRiskGate()
            >>> assessment = gate.assess(
            ...     subagent_type="backend-developer",
            ...     description="Add user authentication",
            ...     prompt="Implement JWT authentication for the API",
            ...     session_context={"previous_failure": True},
            ... )
            >>> assert assessment.level in ValidationLevel
        """
        factors: Dict[str, int] = {}
        triggered_signals: List[str] = []

        # Factor 1: Subagent type risk (0-30 points)
        type_score = self.type_scores.get(subagent_type, 20)
        factors["subagent_type"] = type_score
        if type_score >= 20:
            triggered_signals.append(f"high_risk_type:{subagent_type}")

        # Factor 2: Content risk signals (0-40 points, capped)
        content_score, content_signals = self._analyze_content(description, prompt)
        factors["content_risk"] = min(content_score, 40)
        triggered_signals.extend(content_signals)

        # Factor 3: Prompt complexity (0-20 points)
        complexity_score = self._analyze_complexity(prompt)
        factors["complexity"] = complexity_score

        # Factor 4: Historical context (0-10 points)
        history_score = self._analyze_history(session_context)
        factors["history"] = history_score

        # Compute total score
        total_score = sum(factors.values())

        # Apply strict mode modifier
        if self.strict_mode:
            total_score = min(total_score + 20, 100)
            triggered_signals.append("strict_mode_enabled")

        # Determine validation level and recommendations
        level = self._score_to_level(total_score)
        validators, model = self._get_recommendations(level, triggered_signals)

        return RiskAssessment(
            total_score=total_score,
            level=level,
            factors=factors,
            triggered_signals=triggered_signals,
            recommended_validators=validators,
            recommended_model=model,
        )

    def _analyze_content(
        self, description: str, prompt: str
    ) -> Tuple[int, List[str]]:
        """Analyze content for risk signals using regex patterns.

        Args:
            description: Task description text.
            prompt: Full prompt text.

        Returns:
            Tuple of (score, list of triggered signal names).
        """
        combined = f"{description} {prompt}".lower()
        score = 0
        signals: List[str] = []

        # Check critical patterns (highest risk)
        for pattern, signal_name, points in self.CRITICAL_PATTERNS:
            if re.search(pattern, combined, re.IGNORECASE):
                score += points
                signals.append(f"critical:{signal_name}")

        # Check high-risk patterns
        for pattern, signal_name, points in self.HIGH_PATTERNS:
            if re.search(pattern, combined, re.IGNORECASE):
                score += points
                signals.append(f"high:{signal_name}")

        # Check medium-risk patterns
        for pattern, signal_name, points in self.MEDIUM_PATTERNS:
            if re.search(pattern, combined, re.IGNORECASE):
                score += points
                signals.append(f"medium:{signal_name}")

        return score, signals

    def _analyze_complexity(self, prompt: str) -> int:
        """Analyze prompt complexity.

        Args:
            prompt: Full prompt text.

        Returns:
            Complexity score (0-20).
        """
        score = 0

        # Length-based scoring
        if len(prompt) > 2000:
            score += 10
        elif len(prompt) > 1000:
            score += 5

        # File reference scoring
        file_patterns = re.findall(r"[\w/]+\.\w{2,4}", prompt)
        if len(file_patterns) > 3:
            score += 5

        # Code block detection
        if "```" in prompt or "`" in prompt:
            score += 5

        return min(score, 20)

    def _analyze_history(self, context: Optional[Dict[str, object]]) -> int:
        """Analyze session historical context.

        Args:
            context: Session context dictionary.

        Returns:
            History-based risk score (0-10).
        """
        if not context:
            return 0

        score = 0

        # Previous failure adds risk
        if context.get("previous_failure"):
            score += 5

        # High subagent count indicates complex session
        subagent_count = context.get("subagent_count", 0)
        if isinstance(subagent_count, int) and subagent_count > 5:
            score += 3

        # Recent validation failures are a red flag
        recent_failures = context.get("recent_validation_failures", 0)
        if isinstance(recent_failures, int) and recent_failures > 0:
            score += 5

        return min(score, 10)

    def _score_to_level(self, score: int) -> ValidationLevel:
        """Convert risk score to validation level.

        Args:
            score: Total risk score.

        Returns:
            Appropriate ValidationLevel for the score.
        """
        if score <= self.thresholds["skip_max"]:
            return ValidationLevel.SKIP
        elif score <= self.thresholds["lite_max"]:
            return ValidationLevel.LITE
        elif score <= self.thresholds["standard_max"]:
            return ValidationLevel.STANDARD
        else:
            return ValidationLevel.DEEP

    def _get_recommendations(
        self, level: ValidationLevel, signals: List[str]
    ) -> Tuple[List[str], str]:
        """Get validator and model recommendations for a validation level.

        Args:
            level: The validation level.
            signals: List of triggered risk signals.

        Returns:
            Tuple of (list of validator types, model ID).
        """
        if level == ValidationLevel.SKIP:
            return [], "none"

        elif level == ValidationLevel.LITE:
            return ["correctness"], self.MODEL_HAIKU

        elif level == ValidationLevel.STANDARD:
            validators = ["correctness", "scope"]
            # Add security validator if security-related signals detected
            if any("security" in s or "secret" in s for s in signals):
                validators.append("security")
            return validators, self.MODEL_HAIKU

        else:  # DEEP
            validators = [
                "security",
                "correctness",
                "compliance",
                "scope",
                "groundedness",
            ]
            return validators, self.MODEL_OPUS


def should_validate(
    subagent_type: str,
    description: str,
    prompt: str,
    session_context: Optional[Dict[str, object]] = None,
    config: Optional[Dict[str, object]] = None,
) -> Tuple[bool, RiskAssessment]:
    """Convenience function to check if validation is needed.

    This is the primary entry point for the risk gate. It creates a
    DeterministicRiskGate instance and performs the assessment in one call.

    Args:
        subagent_type: The type/name of the subagent being invoked.
        description: Short description of the task.
        prompt: Full prompt text being sent to the subagent.
        session_context: Optional session-level context dictionary.
        config: Optional configuration for the risk gate.

    Returns:
        Tuple of (should_validate: bool, assessment: RiskAssessment).
        The boolean is True if validation is recommended (level != SKIP).

    Example:
        >>> should_val, assessment = should_validate(
        ...     subagent_type="Explore",
        ...     description="List files in directory",
        ...     prompt="Show me the contents of /src",
        ... )
        >>> if should_val:
        ...     # Spawn validators based on assessment.recommended_validators
        ...     print(f"Validate with: {assessment.recommended_validators}")
        ... else:
        ...     print("Skip validation - low risk operation")
    """
    gate = DeterministicRiskGate(config)
    assessment = gate.assess(subagent_type, description, prompt, session_context)
    return assessment.level != ValidationLevel.SKIP, assessment
