"""SOC2 Audit logging for security-relevant events."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Thread-local storage for audit context
_audit_context = threading.local()


class AuditLogger:
    """
    SOC2-compliant audit logger for security-relevant events.

    Logs events such as:
    - Data access (read/write operations)
    - Configuration changes
    - Authentication attempts
    - Encryption key generation
    """

    _instance: Optional["AuditLogger"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        log_path: Optional[str | Path] = None,
        enabled: bool = True,
    ) -> None:
        """
        Initialize the audit logger.

        Args:
            log_path: Path to audit log file. Defaults to ./audit.log
            enabled: Whether audit logging is enabled
        """
        self.enabled = enabled
        self.log_path = Path(log_path) if log_path else Path("audit.log")
        self._file_lock = threading.Lock()

        if self.enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            # Set restrictive permissions on audit log
            if self.log_path.exists():
                os.chmod(self.log_path, 0o600)

    @classmethod
    def get_instance(cls) -> "AuditLogger":
        """Get or create the singleton audit logger instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    enabled = os.environ.get("ARZULE_AUDIT_LOG_ENABLED", "true").lower() in {
                        "1", "true", "yes", "y"
                    }
                    log_path = os.environ.get("ARZULE_AUDIT_LOG_PATH", "audit.log")
                    cls._instance = cls(log_path=log_path, enabled=enabled)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def log(
        self,
        event_type: str,
        action: str,
        resource: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        actor: Optional[str] = None,
        status: str = "success",
    ) -> None:
        """
        Log an audit event.

        Args:
            event_type: Category of event (e.g., "data_access", "config_change")
            action: Specific action (e.g., "write", "encrypt", "configure")
            resource: Resource being accessed (e.g., file path, endpoint)
            details: Additional context (will be sanitized)
            actor: Identity performing the action (e.g., process ID, user)
            status: Outcome (success, failure, error)
        """
        if not self.enabled:
            return

        # Build audit record
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "action": action,
            "resource": self._sanitize_resource(resource),
            "actor": actor or self._get_default_actor(),
            "status": status,
            "details": self._sanitize_details(details or {}),
        }

        self._write_record(record)

    def _sanitize_resource(self, resource: Optional[str]) -> Optional[str]:
        """Sanitize resource identifiers (remove sensitive path components)."""
        if not resource:
            return None
        # Remove any query parameters or credentials from URLs
        if "://" in resource:
            from urllib.parse import urlparse
            parsed = urlparse(resource)
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        return resource

    def _sanitize_details(self, details: dict[str, Any]) -> dict[str, Any]:
        """Remove sensitive information from details."""
        sensitive_keys = {"password", "secret", "key", "token", "api_key", "authorization"}
        sanitized = {}
        for k, v in details.items():
            if k.lower() in sensitive_keys:
                sanitized[k] = "<redacted>"
            elif isinstance(v, dict):
                sanitized[k] = self._sanitize_details(v)
            else:
                sanitized[k] = v
        return sanitized

    def _get_default_actor(self) -> str:
        """Get default actor identity."""
        return f"pid:{os.getpid()}"

    def _write_record(self, record: dict[str, Any]) -> None:
        """Write audit record to log file."""
        with self._file_lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
            # Ensure restrictive permissions
            os.chmod(self.log_path, 0o600)

    # Convenience methods for common audit events
    def log_data_write(
        self,
        sink_type: str,
        destination: str,
        event_count: int,
        encrypted: bool = False,
    ) -> None:
        """Log a data write operation."""
        self.log(
            event_type="data_access",
            action="write",
            resource=destination,
            details={
                "sink_type": sink_type,
                "event_count": event_count,
                "encrypted": encrypted,
            },
        )

    def log_config_change(
        self,
        setting: str,
        old_value: Any = None,
        new_value: Any = None,
    ) -> None:
        """Log a configuration change."""
        self.log(
            event_type="config_change",
            action="update",
            resource=setting,
            details={
                "old_value": "<redacted>" if "key" in setting.lower() else old_value,
                "new_value": "<redacted>" if "key" in setting.lower() else new_value,
            },
        )

    def log_encryption_key_generated(self, key_path: str) -> None:
        """Log encryption key generation."""
        self.log(
            event_type="security",
            action="key_generated",
            resource=key_path,
            details={"algorithm": "AES-256-Fernet"},
        )

    def log_tls_validation(self, endpoint: str, valid: bool) -> None:
        """Log TLS validation result."""
        self.log(
            event_type="security",
            action="tls_validation",
            resource=endpoint,
            status="success" if valid else "failure",
        )


# Module-level convenience function
def audit_log() -> AuditLogger:
    """Get the audit logger instance."""
    return AuditLogger.get_instance()





