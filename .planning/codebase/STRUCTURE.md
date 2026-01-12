# Codebase Structure

**Analysis Date:** 2026-01-10

## Directory Layout

```
arzuleIngestionWrapper/
├── src/
│   └── arzule_ingest/        # Main SDK package
│       ├── __init__.py       # Public API: init(), new_run(), etc.
│       ├── run.py            # ArzuleRun context manager
│       ├── config.py         # Configuration dataclass
│       ├── cli.py            # CLI commands
│       ├── logger.py         # Centralized logging
│       ├── audit.py          # SOC2 audit logging
│       ├── sanitize.py       # PII/secret redaction
│       ├── ids.py            # ID generation utilities
│       ├── run_managers/     # Context management
│       ├── sinks/            # Telemetry sinks
│       ├── blobs/            # Blob storage
│       ├── crewai/           # CrewAI integration
│       ├── langchain/        # LangChain integration
│       ├── langgraph/        # LangGraph integration
│       ├── autogen/          # AutoGen v0.2 integration
│       ├── autogen_v2/       # AutoGen v0.7+ integration
│       ├── claude/           # Claude Code integration
│       ├── scenarios/        # Test scenarios (excluded from build)
│       ├── demo/             # Demo code (excluded from build)
│       └── forensics/        # Forensic tools (excluded from build)
├── tests/                    # Test files
├── pyproject.toml            # Project configuration
├── README.md                 # Documentation
├── LICENSE                   # MIT license
└── .gitignore                # Git ignore rules
```

## Directory Purposes

**src/arzule_ingest/**
- Purpose: Main SDK package
- Contains: Core runtime, framework integrations, utilities
- Key files: `__init__.py`, `run.py`, `config.py`, `cli.py`

**src/arzule_ingest/run_managers/**
- Purpose: Context management components
- Contains: Span, task, async, and agent context managers
- Key files: `span_manager.py`, `task_manager.py`, `async_tracker.py`, `agent_context.py`

**src/arzule_ingest/sinks/**
- Purpose: Telemetry sink implementations
- Contains: HTTP batch, file JSONL, multi-sink
- Key files: `base.py`, `http_batch.py`, `file_jsonl.py`, `multi.py`

**src/arzule_ingest/blobs/**
- Purpose: Blob storage abstractions
- Contains: Inline and HTTP blob references
- Key files: `base.py`, `inline.py`, `http_blob.py`

**src/arzule_ingest/crewai/**
- Purpose: CrewAI framework integration
- Contains: Listener, hooks, normalization, handoff detection
- Key files: `install.py`, `listener.py`, `hooks_tool.py`, `hooks_llm.py`, `normalize.py`, `handoff.py`

**src/arzule_ingest/langchain/**
- Purpose: LangChain framework integration
- Contains: Callback handler, normalization, handoff detection
- Key files: `install.py`, `callback_handler.py`, `normalize.py`, `handoff.py`

**src/arzule_ingest/langgraph/**
- Purpose: LangGraph framework integration
- Contains: Callback handler for graph/node execution
- Key files: `install.py`, `callback_handler.py`, `normalize.py`

**src/arzule_ingest/autogen/**
- Purpose: AutoGen v0.2 (pyautogen) integration
- Contains: Message and LLM hooks
- Key files: `install.py`, `hooks.py`, `normalize.py`

**src/arzule_ingest/autogen_v2/**
- Purpose: AutoGen v0.7+ (autogen-core) integration
- Contains: New telemetry hooks
- Key files: `install.py`, `hooks.py`, `telemetry.py`, `normalize.py`

**src/arzule_ingest/claude/**
- Purpose: Claude Code integration
- Contains: Hook handler, session management, transcript parsing
- Key files: `hook.py`, `wrapper.py`, `install.py`, `session.py`, `turn.py`, `transcript.py`, `normalize.py`, `security.py`

**tests/**
- Purpose: Test files
- Contains: Unit and integration tests
- Key files: `test_sanitize.py`, `test_seq_monotonic.py`, `test_soc2_compliance.py`, `test_forensics.py`

## Key File Locations

**Entry Points:**
- `src/arzule_ingest/__init__.py` - SDK public API
- `src/arzule_ingest/cli.py` - CLI entry (`arzule` command)
- `src/arzule_ingest/claude/wrapper.py` - Claude wrapper (`arzule-claude`)
- `src/arzule_ingest/claude/install.py` - Hook installer (`arzule-claude-install`)

**Configuration:**
- `pyproject.toml` - Project metadata, dependencies
- `src/arzule_ingest/config.py` - ArzuleConfig dataclass
- `.env` - Development environment (gitignored)
- `~/.arzule/config` - User configuration (runtime)

**Core Logic:**
- `src/arzule_ingest/run.py` - ArzuleRun context manager
- `src/arzule_ingest/run_managers/span_manager.py` - Span hierarchy
- `src/arzule_ingest/sinks/http_batch.py` - HTTP delivery
- `src/arzule_ingest/sanitize.py` - PII/secret redaction

**Security:**
- `src/arzule_ingest/audit.py` - SOC2 audit logging
- `src/arzule_ingest/sanitize.py` - Secret redaction
- `src/arzule_ingest/claude/security.py` - HMAC validation

**Testing:**
- `tests/test_sanitize.py` - Sanitization tests
- `tests/test_seq_monotonic.py` - Sequence tests
- `tests/test_soc2_compliance.py` - Compliance tests

## Naming Conventions

**Files:**
- snake_case for all Python modules: `span_manager.py`, `callback_handler.py`
- `test_*.py` for test files
- `__init__.py` for package initialization

**Directories:**
- snake_case for all directories
- Framework names: `crewai/`, `langchain/`, `autogen_v2/`
- Functional groupings: `sinks/`, `blobs/`, `run_managers/`

**Special Patterns:**
- `install.py` - Framework instrumentation entry point
- `normalize.py` - Event normalization to TraceEvent schema
- `hooks.py` / `callback_handler.py` - Event capture
- `handoff.py` - Handoff detection logic
- `spanctx.py` - Span context management

## Where to Add New Code

**New Framework Integration:**
- Primary code: `src/arzule_ingest/{framework_name}/`
- Required files: `install.py`, `normalize.py`, `hooks.py` or `callback_handler.py`
- Optional: `handoff.py`, `spanctx.py`
- Export from: `src/arzule_ingest/__init__.py`

**New Sink Type:**
- Implementation: `src/arzule_ingest/sinks/{name}.py`
- Inherit from: `TelemetrySink` base class
- Tests: `tests/test_{name}_sink.py`

**New CLI Command:**
- Add to: `src/arzule_ingest/cli.py`
- Follow pattern of existing commands

**New Run Manager:**
- Implementation: `src/arzule_ingest/run_managers/{name}.py`
- Integrate with: `src/arzule_ingest/run.py`

**Utilities:**
- Shared helpers: `src/arzule_ingest/` (root of package)
- Type definitions: Inline with modules (no separate types file)

## Special Directories

**scenarios/**
- Purpose: Test scenarios for development
- Source: Created during development/testing
- Committed: No (excluded from build via pyproject.toml)

**demo/**
- Purpose: Demo and example code
- Source: Created for documentation/testing
- Committed: No (excluded from build)

**forensics/**
- Purpose: Internal debugging tools
- Source: Development utilities
- Committed: No (excluded from build)

---

*Structure analysis: 2026-01-10*
*Update when directory structure changes*
