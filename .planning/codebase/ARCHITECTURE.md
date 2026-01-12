# Architecture

**Analysis Date:** 2026-01-10

## Pattern Overview

**Overall:** Multi-Framework Instrumentation SDK with Plugin Adapters

**Key Characteristics:**
- Modular plugin-based design with framework adapters
- Context-managed run tracking with hierarchical spans
- Multi-destination telemetry sinks
- Thread-safe concurrency with ContextVar + global registry fallback

## Layers

```
┌─────────────────────────────────────────────────────────┐
│          Public API Layer (init, new_run, etc)          │
│          src/arzule_ingest/__init__.py                  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│         Framework Integration Layer (Adapters)          │
│  crewai/ | langchain/ | langgraph/ | autogen/          │
│  autogen_v2/ | claude/                                  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│        Core Telemetry Layer (Trace Management)          │
│  ArzuleRun + SpanManager + TaskManager + AgentContext   │
│  src/arzule_ingest/run.py + run_managers/               │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│          Data Processing & Sink Layer                   │
│  sanitize.py | blobs/ | sinks/ (http_batch, file)      │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│    Infrastructure Layer (CLI, Config, Audit)            │
│  cli.py | config.py | audit.py | logger.py             │
└─────────────────────────────────────────────────────────┘
```

**Public API Layer:**
- Purpose: User-facing SDK entry points
- Contains: `init()`, `new_run()`, `ensure_run()`, `shutdown()`, `current_run()`
- Location: `src/arzule_ingest/__init__.py`
- Depends on: Core telemetry, framework adapters
- Used by: Application code

**Framework Integration Layer:**
- Purpose: Framework-specific instrumentation adapters
- Contains: install.py, normalize.py, hooks/callback handlers per framework
- Location: `src/arzule_ingest/{crewai,langchain,langgraph,autogen,autogen_v2,claude}/`
- Depends on: Core telemetry layer, framework libraries (optional)
- Used by: Public API (auto-detected) or direct instrumentation calls

**Core Telemetry Layer:**
- Purpose: Run/span/trace management and context propagation
- Contains: `ArzuleRun` context manager, span/task/agent managers
- Location: `src/arzule_ingest/run.py`, `src/arzule_ingest/run_managers/`
- Depends on: Data processing layer
- Used by: Framework adapters

**Data Processing Layer:**
- Purpose: Event serialization, sanitization, and delivery
- Contains: Telemetry sinks, blob storage, PII redaction
- Location: `src/arzule_ingest/sinks/`, `src/arzule_ingest/blobs/`, `src/arzule_ingest/sanitize.py`
- Depends on: Infrastructure layer
- Used by: Core telemetry layer

**Infrastructure Layer:**
- Purpose: CLI, configuration, audit logging, utilities
- Contains: CLI commands, config parsing, audit trail, ID generation
- Location: `src/arzule_ingest/cli.py`, `src/arzule_ingest/config.py`, `src/arzule_ingest/audit.py`
- Depends on: Python stdlib, cryptography
- Used by: All layers

## Data Flow

**Initialization Flow:**
1. User calls `arzule_ingest.init()` → `src/arzule_ingest/__init__.py`
2. Creates `HttpBatchSink` with API credentials → `src/arzule_ingest/sinks/http_batch.py`
3. Auto-detects installed frameworks (CrewAI, LangChain, AutoGen, LangGraph)
4. Instruments each framework with hooks/callbacks/listeners
5. Registers cleanup handler with `atexit`

**Run Context Flow:**
1. First framework event triggers `ensure_run()` → creates `ArzuleRun` context
2. `ArzuleRun.__enter__()`:
   - Generates `run_id`, `trace_id`, root `span_id`
   - Initializes `SpanManager`, `TaskManager`, `AsyncTracker`, `AgentContext`
   - Registers run in global registry (thread-safe fallback)
   - Emits `run.start` event
3. Framework event arrives → instrumentation hook captures it
4. Hook creates trace event dict via `_make_event()`
5. Event emitted via `run.emit()` → `sink.write()`
6. Events batched in memory buffer
7. On `flush()` or batch full: HTTP POST to endpoint
8. `ArzuleRun.__exit__()`: Emits `run.end`, flushes remaining events

**Span Hierarchy:**
```
run.start (root_span_id)
  ├─ crew.start (crew_span_id)
  │   ├─ task.start (task_span_id)
  │   │   ├─ agent.start (agent_span_id)
  │   │   │   ├─ tool.start
  │   │   │   │   └─ tool.end
  │   │   │   └─ agent.end
  │   │   ├─ handoff.proposed
  │   │   └─ task.end
  │   └─ crew.end
  └─ run.end
```

**State Management:**
- ContextVar (`_active_run`) for async context propagation
- Global registry (`_run_registry`) for thread-safe fallback
- File-based state: None (SDK is stateless)

## Key Abstractions

**ArzuleRun (Context Manager):**
- Purpose: Central orchestrator for a single observability run
- Location: `src/arzule_ingest/run.py`
- Pattern: Context manager with thread-safe sequence numbering
- Examples: `with ArzuleRun(...) as run:`, `run.emit(event)`

**TelemetrySink (Strategy Pattern):**
- Purpose: Abstract interface for event delivery
- Location: `src/arzule_ingest/sinks/base.py`
- Pattern: Strategy pattern with pluggable implementations
- Implementations:
  - `HttpBatchSink` → `src/arzule_ingest/sinks/http_batch.py`
  - `JsonlFileSink` → `src/arzule_ingest/sinks/file_jsonl.py`
  - `MultiSink` → `src/arzule_ingest/sinks/multi.py`

**Framework Adapters (Template Pattern):**
- Purpose: Framework-specific instrumentation
- Pattern: Each framework has: install.py, normalize.py, hooks/callback_handler.py
- Structure:
  - `install.py`: Sets up instrumentation (called once)
  - `normalize.py`: Converts framework events to TraceEvent schema
  - `hooks.py` / `callback_handler.py`: Captures events

**Run Managers (Delegation):**
- Purpose: Manage specific aspects of run context
- Location: `src/arzule_ingest/run_managers/`
- Components:
  - `SpanManager`: Stack-based parent/child span tracking
  - `TaskManager`: Task-to-span mapping for concurrent execution
  - `AsyncTracker`: Async task completion with timeouts
  - `AgentContext`: Thread-local agent state tracking

## Entry Points

**Python SDK:**
- Location: `src/arzule_ingest/__init__.py`
- Triggers: Import and call `init()`, `new_run()`, etc.
- Responsibilities: Initialize SDK, manage runs, auto-instrument frameworks

**CLI Tool:**
- Location: `src/arzule_ingest/cli.py:main()`
- Triggers: `arzule` command
- Responsibilities: Configure credentials, view traces, setup integration

**Claude Wrapper:**
- Location: `src/arzule_ingest/claude/wrapper.py:main()`
- Triggers: `arzule-claude` command
- Responsibilities: Wrap Claude Code with instrumentation

**Claude Install:**
- Location: `src/arzule_ingest/claude/install.py:main()`
- Triggers: `arzule-claude-install` command
- Responsibilities: Install hooks in `.claude/settings.json`

**Hook Handler:**
- Location: `src/arzule_ingest/claude/__main__.py`
- Triggers: Called by Claude Code via hook system
- Responsibilities: Process hook events, emit telemetry

## Error Handling

**Strategy:** Throw exceptions at boundaries, fail gracefully in instrumentation

**Patterns:**
- Custom exceptions: `TLSRequiredError` in `src/arzule_ingest/sinks/http_batch.py`
- Validation: UUID format, required env vars in `init()`
- Framework errors: Caught silently to avoid breaking user's application
- Critical comments: `# CRITICAL:` prefix for audit trail

## Cross-Cutting Concerns

**Logging:**
- Centralized: `src/arzule_ingest/logger.py`
- Format: `[arzule] timestamp LEVEL message`
- Output: stderr
- Debug: `~/.arzule/hook_debug.log`

**Validation:**
- Config validation in `init()` function
- UUID format validation for tenant_id, project_id
- TLS enforcement for HTTP endpoints

**Security:**
- PII redaction: `src/arzule_ingest/sanitize.py`
- Secret detection: API keys, tokens, credit cards
- Audit logging: `src/arzule_ingest/audit.py` (SOC2 compliance)
- HMAC validation: `src/arzule_ingest/claude/security.py`

**Threading & Concurrency:**
- ContextVar: `_active_run` for async context propagation
- Global registry: `_run_registry` with lock for thread-safe fallback
- Locks: `_run_lock`, `_seq_lock`, `_agent_span_lock`
- Double-check locking: Singleton patterns

---

*Architecture analysis: 2026-01-10*
*Update when major patterns change*
