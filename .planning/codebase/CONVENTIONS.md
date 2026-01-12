# Coding Conventions

**Analysis Date:** 2026-01-10

## Naming Patterns

**Files:**
- snake_case for all modules: `span_manager.py`, `callback_handler.py`, `hooks_tool.py`
- `test_*.py` for test files: `test_sanitize.py`, `test_seq_monotonic.py`
- `__init__.py` for package initialization

**Functions:**
- snake_case for all functions: `new_run_id()`, `get_logger()`, `redact_secrets()`
- Leading underscore for private: `_safe_getattr()`, `_compute_input_hash()`, `_check_crewai_available()`
- No special prefix for async functions

**Variables:**
- snake_case for variables: `event_type`, `span_id`, `trace_id`
- UPPER_SNAKE_CASE for constants: `REDACT_KEYS`, `SECRET_PATTERNS`, `PII_PATTERNS`, `DEFAULT_INGEST_URL`
- No underscore prefix for private attributes (Python convention)

**Types:**
- PascalCase for classes: `ArzuleRun`, `ArzuleConfig`, `HttpBatchSink`, `TelemetrySink`
- PascalCase for type aliases (when used)
- No `I` prefix for interfaces

**Modules/Packages:**
- Framework names: `crewai/`, `langchain/`, `autogen/`, `autogen_v2/`
- Functional groupings: `sinks/`, `blobs/`, `run_managers/`

## Code Style

**Formatting:**
- 4-space indentation (Python standard)
- Line length: ~100 characters (soft limit)
- Double quotes for docstrings: `"""..."""`
- Mixed quotes for strings (no strict enforcement)
- Trailing newlines enforced (POSIX compliance)

**Linting:**
- No explicit linter config found
- Follows PEP 8 conventions
- Type hints throughout (Python 3.10+ syntax)

**Python Version Features:**
- `from __future__ import annotations` in all files
- Union syntax: `tuple[bool, str]`, `dict[str, Any]`
- Optional syntax: `Optional[str]` from typing module
- Requires Python 3.10+ (`pyproject.toml`)

## Import Organization

**Order:**
1. Future imports: `from __future__ import annotations`
2. Standard library: `import json`, `import threading`, `import sys`
3. Third-party packages: `import httpx`, `from pydantic import...`
4. Local imports: `from .run import ArzuleRun`, `from ..ids import new_span_id`

**Grouping:**
- Blank line between groups
- Alphabetical within groups (not strictly enforced)
- Type imports mixed with regular imports

**Path Aliases:**
- None configured
- Relative imports: `from .`, `from ..`

## Error Handling

**Patterns:**
- Custom exceptions for domain errors: `TLSRequiredError` in `src/arzule_ingest/sinks/http_batch.py`
- ValueError for validation errors with descriptive messages
- Try-except for optional imports (framework detection)

**Error Types:**
- Throw on: invalid input, missing required config, security violations
- Catch silently in: instrumentation hooks (to avoid breaking user's app)
- Log and continue in: non-critical framework operations

**Exception Handling:**
- No bare `except:` clauses
- All exceptions properly typed
- Use `raise ... from e` for exception chaining

## Logging

**Framework:**
- Custom logger: `src/arzule_ingest/logger.py`
- Uses stdlib `logging` module
- Output: stderr (StreamHandler)

**Patterns:**
- Format: `[arzule] YYYY-MM-DD HH:MM:SS LEVEL message`
- Levels: DEBUG, INFO, WARNING, ERROR
- Default level: INFO
- propagate=False to prevent double logging

**Where to Log:**
- Entry/exit of main operations
- Error conditions
- Debug info in hook handlers: `~/.arzule/hook_debug.log`

## Comments

**When to Comment:**
- Explain "why" not "what"
- Document business logic and edge cases
- Critical sections: `# CRITICAL:` prefix for security/race condition notes
- Algorithm explanations where not obvious

**Docstrings:**
- Triple-quoted docstrings for all public functions/classes
- Google-style format with Args, Returns, Raises sections
- Module docstrings at top of each file

**Example:**
```python
def init(
    api_key: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> dict:
    """
    Initialize Arzule with minimal configuration.

    Args:
        api_key: API key for authentication. Defaults to ARZULE_API_KEY env var.
        tenant_id: Tenant ID. Defaults to ARZULE_TENANT_ID env var.

    Returns:
        Config dict with tenant_id, project_id for reference.

    Raises:
        ValueError: If required configuration is missing.
    """
```

**TODO Comments:**
- Format: `# TODO: description`
- No username (use git blame)
- Link to issue if exists

## Function Design

**Size:**
- Keep under 50-100 lines (soft limit)
- Extract helpers for complex logic
- Some large functions exist in `hook.py` (1,919 lines total)

**Parameters:**
- Type hints on all parameters
- Optional parameters with defaults
- Use dataclasses for complex parameter objects

**Return Values:**
- Explicit return type hints
- Return early for guard clauses
- Use `Optional[T]` or `T | None` for nullable returns

## Module Design

**Exports:**
- Named exports preferred
- `__all__` list in `__init__.py` defines public API
- No default exports (Python doesn't have them)

**Barrel Files:**
- `__init__.py` re-exports public API
- Framework modules have minimal exports
- Internal helpers stay private

**Organization:**
- One main class/concept per file
- Related utilities in same file
- Avoid circular imports with `TYPE_CHECKING`

## Threading & Concurrency

**Patterns:**
- ContextVar for async context propagation: `_active_run`
- threading.Lock for critical sections: `_run_lock`, `_seq_lock`
- Global registry with lock for thread-safe fallback
- Double-check locking for singleton initialization

**Best Practices:**
- Use `with lock:` context manager
- Document thread-safety in docstrings
- Mark thread-safe methods explicitly
- Use `threading.local()` for thread-local storage

## Security Conventions

**Secrets:**
- Never log secrets (redact first)
- Use environment variables for credentials
- Validate TLS by default
- Audit log security events

**PII:**
- Redact by default (configurable)
- Pattern-based detection
- Key-based redaction for sensitive fields

---

*Convention analysis: 2026-01-10*
*Update when patterns change*
