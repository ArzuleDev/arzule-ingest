# Testing Patterns

**Analysis Date:** 2026-01-10

## Test Framework

**Runner:**
- pytest >= 8.0.0
- Config: `pyproject.toml` `[tool.pytest.ini_options]`

**Assertion Library:**
- pytest built-in assert
- Matchers: `assert`, `assert ... in`, `pytest.raises()`

**Run Commands:**
```bash
pytest tests/                          # Run all tests
pytest tests/test_sanitize.py          # Run specific file
pytest tests/test_sanitize.py::TestRedactSecrets  # Run specific class
pytest -v                              # Verbose output
```

## Test File Organization

**Location:**
- `tests/` directory at repository root
- Not co-located with source files

**Naming:**
- `test_*.py` for all test files
- No distinction between unit/integration in filename

**Structure:**
```
tests/
├── __init__.py
├── test_forensics.py           # 634 lines - Forensic analysis tests
├── test_handoff_key_injection.py  # 160 lines - Handoff logic tests
├── test_sanitize.py            # 234 lines - PII/secret redaction tests
├── test_seq_monotonic.py       # 171 lines - Sequence number tests
└── test_soc2_compliance.py     # 228 lines - SOC2 compliance tests
```

## Test Structure

**Suite Organization:**
```python
"""Tests for payload sanitization."""

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

    def test_preserves_normal_text(self):
        """Test that normal text is preserved."""
        text = "This is normal text without secrets"
        result = redact_secrets(text)
        assert result == text
```

**Patterns:**
- Class-based test organization by feature
- Descriptive method names with `test_` prefix
- Docstrings describe expected behavior
- No setup/teardown methods (inline setup)
- Arrange/Act/Assert pattern

## Mocking

**Framework:**
- No dedicated mocking library detected
- Inline mock classes defined in test files

**Patterns:**
```python
class MockSink(TelemetrySink):
    """Mock sink for testing."""

    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(event)

    def flush(self):
        pass
```

**What to Mock:**
- TelemetrySink for testing event emission
- No external API mocking (tests run against real code)

**What NOT to Mock:**
- Pure functions
- Internal business logic
- Sanitization functions (tested directly)

## Fixtures and Factories

**Test Data:**
```python
# Inline test data in test methods
def test_redacts_email(self):
    text = "Contact: john.doe@example.com for more info"
    result = redact_pii(text)
    assert "john.doe@example.com" not in result
```

**Location:**
- No separate fixtures directory
- Mock classes defined at module level in test files
- Test data created inline

## Coverage

**Requirements:**
- No enforced coverage target
- Coverage tracked for awareness

**Configuration:**
- pytest-cov not detected in dependencies
- Run manually: `pytest --cov=arzule_ingest`

## Test Types

**Unit Tests:**
- `tests/test_sanitize.py` - Secret/PII redaction, truncation
- `tests/test_seq_monotonic.py` - Sequence number generation, ID uniqueness
- Test single functions in isolation

**Integration Tests:**
- `tests/test_soc2_compliance.py` - TLS enforcement, audit logging
- `tests/test_handoff_key_injection.py` - Handoff detection logic
- Test multiple modules together

**Forensic Tests:**
- `tests/test_forensics.py` - Trace analysis, event reconstruction
- Larger integration scenarios

**E2E Tests:**
- Not currently implemented
- Framework integration tested via scenarios (excluded from build)

## Common Patterns

**Async Testing:**
```python
# pytest-asyncio available but not heavily used
# Most tests are synchronous
```

**Error Testing:**
```python
def test_rejects_http_url_by_default(self):
    """HTTP URLs should be rejected when require_tls=True (default)."""
    with pytest.raises(TLSRequiredError) as exc_info:
        HttpBatchSink(
            endpoint_url="http://example.com/ingest",
            api_key="test-key",
        )
    assert "SOC2 compliance requires HTTPS" in str(exc_info.value)
```

**Context Manager Testing:**
```python
def test_seq_starts_at_1(self):
    """Test that sequence numbers start at 1."""
    sink = MockSink()
    with ArzuleRun(tenant_id="t1", project_id="p1", sink=sink) as run:
        pass
    assert sink.events[0]["seq"] == 1
```

**JSON Serialization Testing:**
```python
def test_event_is_json_serializable(self):
    """Test that events can be serialized to JSON."""
    # ...
    for event in sink.events:
        json_str = json.dumps(event)
        parsed = json.loads(json_str)
        assert parsed == event
```

## Test Categories by Class

**TestRedactSecrets (4 tests)** - `test_sanitize.py`
- API key redaction
- Bearer token redaction
- OpenAI key redaction
- Normal text preservation

**TestRedactPii (10+ tests)** - `test_sanitize.py`
- Email redaction
- Phone number redaction
- Credit card redaction
- SSN redaction
- Various PII patterns

**TestSanitize (6 tests)** - `test_sanitize.py`
- Combined sanitization
- Nested structure handling
- Depth limiting

**TestTruncateString (3 tests)** - `test_sanitize.py`
- String truncation
- Boundary conditions

**TestTLSEnforcement (3 tests)** - `test_soc2_compliance.py`
- HTTP rejection
- HTTPS acceptance
- TLS bypass option

**TestSeqMonotonic (3 tests)** - `test_seq_monotonic.py`
- Sequence start value
- Monotonic increase
- Uniqueness

**TestRunIds (3 tests)** - `test_seq_monotonic.py`
- ID format validation
- Uniqueness

**TestEventSchema (2 tests)** - `test_seq_monotonic.py`
- Schema validation
- JSON serialization

---

*Testing analysis: 2026-01-10*
*Update when test patterns change*
