# External Integrations

**Analysis Date:** 2026-01-10

## APIs & External Services

**Arzule Ingest API:**
- Primary backend service for telemetry data
  - SDK/Client: httpx (async HTTP client)
  - Auth: Bearer token via `ARZULE_API_KEY` env var
  - Endpoint: `https://ingest.arzule.com/ingest` (default)
  - Protocol: HTTP POST with NDJSON payload
  - TLS: Required by default for SOC2 compliance (`src/arzule_ingest/sinks/http_batch.py`)

**No other external APIs:**
- This is an SDK that instruments other frameworks
- Does not call OpenAI, Anthropic, or other LLM APIs directly

## Data Storage

**Databases:**
- None - Stateless SDK, no persistent storage

**File Storage:**
- Local JSONL sink option: `src/arzule_ingest/sinks/file_jsonl.py`
- Output directory: `out/` (gitignored)
- Debug logs: `~/.arzule/hook_debug.log`

**Caching:**
- None - All state is in-memory per-run

## Authentication & Identity

**Auth Provider:**
- API key authentication (Bearer token)
  - Token storage: Environment variable `ARZULE_API_KEY`
  - User config: `~/.arzule/config` (key=value file)
  - No OAuth or session management

**No OAuth Integrations:**
- SDK does not authenticate users
- Operates on behalf of the application

## Monitoring & Observability

**Error Tracking:**
- None configured externally
  - Internal: `src/arzule_ingest/logger.py` logs to stderr
  - Debug logs: `~/.arzule/hook_debug.log`

**Analytics:**
- None - This IS the analytics collection SDK

**Logs:**
- Centralized logger: `src/arzule_ingest/logger.py`
- Format: `[arzule] YYYY-MM-DD HH:MM:SS LEVEL message`
- Output: stderr by default

**Audit Logging (SOC2 Compliance):**
- `src/arzule_ingest/audit.py` - Security event tracking
- Events: data access, config changes, encryption key generation
- File: `audit.log` (restrictive 0o600 permissions)

## CI/CD & Deployment

**Hosting:**
- PyPI package deployment
- Runs on user's infrastructure

**CI Pipeline:**
- Not detected in this repository
- Likely external (GitHub Actions, GitLab CI)

## Environment Configuration

**Development:**
- Required env vars: `ARZULE_API_KEY`, `ARZULE_TENANT_ID`, `ARZULE_PROJECT_ID`
- Secrets location: `.env` file (gitignored)
- No mock services - connects to real API or writes to local files

**Production:**
- Same env vars required
- TLS enforced by default (configurable via `require_tls` parameter)
- Secrets via environment variables or user config file

## Webhooks & Callbacks

**Incoming:**
- Claude Code hooks - JSON via stdin
  - Events: SessionStart, SessionEnd, UserPromptSubmit, Stop, PreToolUse, PostToolUse, SubagentStart, SubagentStop, PreCompact, Notification
  - Handler: `src/arzule_ingest/claude/hook.py`
  - Verification: HMAC signature validation (`src/arzule_ingest/claude/security.py`)

**Outgoing:**
- HTTP POST to Arzule Ingest API
  - Batching: 100 events default, 5-second flush interval
  - Format: NDJSON (newline-delimited JSON)
  - Retry: None currently - fire and forget

## Multi-Agent Framework Integrations

**CrewAI Integration:**
- Event Listener: `src/arzule_ingest/crewai/listener.py`
- Hook Points: crew/agent/task lifecycle, tool calls, LLM calls
- Handoff Detection: `src/arzule_ingest/crewai/handoff.py`, `src/arzule_ingest/crewai/implicit_handoff.py`

**LangChain/LangGraph Integration:**
- Callback Handler: `src/arzule_ingest/langchain/callback_handler.py`
- Hook Points: LLM calls, chain execution, tool invocations, agent actions, retriever calls
- LangGraph Handler: `src/arzule_ingest/langgraph/callback_handler.py` (graph/node execution)

**AutoGen Integration:**
- v0.2 (Legacy): `src/arzule_ingest/autogen/hooks.py`
- v0.7+ (New): `src/arzule_ingest/autogen_v2/hooks.py`, `src/arzule_ingest/autogen_v2/telemetry.py`
- Auto-detection: Detects installed version and instruments accordingly

**Claude Code Integration:**
- Hook Handler: `src/arzule_ingest/claude/hook.py`
- Session Management: `src/arzule_ingest/claude/session.py`, `src/arzule_ingest/claude/turn.py`
- Transcript Analysis: `src/arzule_ingest/claude/transcript.py`
- Wrapper CLI: `src/arzule_ingest/claude/wrapper.py`

## Data Transport & Serialization

**HTTP Batch Sink:**
- Location: `src/arzule_ingest/sinks/http_batch.py`
- Batching: Configurable (default 100 events, 5s flush)
- Background flush thread for async delivery
- JSON serialization with `default=str` for non-serializable objects

**Sanitization:**
- PII Redaction: `src/arzule_ingest/sanitize.py`
- Secret Detection: API keys, tokens, credit cards
- Payload Limits: 64KB max inline, 20K chars max value

**Security Features:**
- TLS enforcement (SOC2 compliant)
- HMAC validation for Claude Code hooks
- AES-256-Fernet encryption for audit logs
- PII redaction enabled by default

---

*Integration audit: 2026-01-10*
*Update when adding/removing external services*
