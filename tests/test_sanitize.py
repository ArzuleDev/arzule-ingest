"""Tests for payload sanitization."""

import pytest

from arzule_ingest.sanitize import (
    redact_pii,
    redact_secrets,
    sanitize,
    truncate_string,
)


class TestRedactSecrets:
    """Tests for secret redaction."""

    def test_redacts_api_key(self):
        """Test that API keys are redacted."""
        text = "api_key=sk-1234567890abcdef"
        result = redact_secrets(text)
        assert "sk-1234567890abcdef" not in result
        assert "[REDACTED]" in result

    def test_redacts_bearer_token(self):
        """Test that Bearer tokens are redacted."""
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        result = redact_secrets(text)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result
        assert "[REDACTED]" in result

    def test_redacts_openai_key(self):
        """Test that OpenAI-style keys are redacted."""
        text = "Using key sk-abcdefghijklmnopqrstuvwxyz123456"
        result = redact_secrets(text)
        assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in result

    def test_preserves_normal_text(self):
        """Test that normal text is preserved."""
        text = "This is normal text without secrets"
        result = redact_secrets(text)
        assert result == text


class TestRedactPii:
    """Tests for PII redaction."""

    def test_redacts_email(self):
        """Test that emails are redacted."""
        text = "Contact: john.doe@example.com for more info"
        result = redact_pii(text)
        assert "john.doe@example.com" not in result
        assert "[PII_REDACTED]" in result

    def test_redacts_phone(self):
        """Test that phone numbers are redacted."""
        text = "Call me at 555-123-4567"
        result = redact_pii(text)
        assert "555-123-4567" not in result
        assert "[PII_REDACTED]" in result

    def test_preserves_normal_text(self):
        """Test that normal text is preserved."""
        text = "This is normal text without PII"
        result = redact_pii(text)
        assert result == text


class TestSanitize:
    """Tests for recursive sanitization."""

    def test_sanitizes_dict_with_secret_keys(self):
        """Test that secret keys are redacted in dicts."""
        payload = {
            "user": "john",
            "api_key": "secret123",
            "data": [1, 2, 3],
        }
        result = sanitize(payload, redact=True)

        assert result["user"] == "john"
        assert result["api_key"] == "<redacted>"
        assert result["data"] == [1, 2, 3]

    def test_truncates_long_strings(self):
        """Test that long strings are truncated."""
        payload = {"data": "x" * 30000}
        result = sanitize(payload, max_chars=1000)

        assert len(result["data"]) < 30000
        assert "truncated" in result["data"]

    def test_handles_nested_dicts(self):
        """Test that nested dicts are sanitized."""
        payload = {
            "outer": {
                "password": "secret",
                "value": 123,
            }
        }
        result = sanitize(payload, redact=True)

        assert result["outer"]["password"] == "<redacted>"
        assert result["outer"]["value"] == 123

    def test_handles_lists(self):
        """Test that lists are sanitized."""
        payload = [{"token": "abc"}, {"value": 123}]
        result = sanitize(payload, redact=True)

        assert result[0]["token"] == "<redacted>"
        assert result[1]["value"] == 123

    def test_handles_bytes(self):
        """Test that bytes are converted to placeholder."""
        payload = {"data": b"binary data"}
        result = sanitize(payload)

        assert result["data"] == "<bytes:11>"

    def test_handles_depth_limit(self):
        """Test that deep nesting is handled."""
        # Create deeply nested structure
        payload: dict = {"level": 0}
        current = payload
        for i in range(20):
            current["nested"] = {"level": i + 1}
            current = current["nested"]

        result = sanitize(payload)

        # Should not raise, should have max_depth marker
        assert "<max_depth>" in str(result)


class TestTruncateString:
    """Tests for string truncation."""

    def test_short_string_unchanged(self):
        """Test that short strings are unchanged."""
        s = "short"
        result = truncate_string(s, max_len=100)
        assert result == s

    def test_long_string_truncated(self):
        """Test that long strings are truncated."""
        s = "a" * 300
        result = truncate_string(s, max_len=100)

        assert len(result) == 100
        assert result.endswith("...")

    def test_exact_length_unchanged(self):
        """Test that exact-length strings are unchanged."""
        s = "a" * 100
        result = truncate_string(s, max_len=100)
        assert result == s

