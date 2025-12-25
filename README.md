# arzule-ingest

Lightweight SDK for capturing multi-agent traces and sending them to Arzule.

## Installation

```bash
# Core SDK only (no framework dependencies)
pip install arzule-ingest

# With CrewAI integration
pip install arzule-ingest[crewai]
```

## Quick Start

### Option 1: One-line setup (recommended)

```python
import arzule_ingest

# Initialize with environment variables
arzule_ingest.init()

# Your CrewAI code runs as normal - traces are captured automatically
result = crew.kickoff(inputs={...})
```

Required environment variables:
- `ARZULE_API_KEY` - Your API key
- `ARZULE_TENANT_ID` - Your tenant ID
- `ARZULE_PROJECT_ID` - Your project ID

### Option 2: Explicit configuration

```python
import arzule_ingest

arzule_ingest.init(
    api_key="your-api-key",
    tenant_id="your-tenant-id",
    project_id="your-project-id",
)

result = crew.kickoff(inputs={...})
```

### Option 3: Manual instrumentation

```python
from arzule_ingest import ArzuleRun
from arzule_ingest.sinks import JsonlFileSink, HttpBatchSink
from arzule_ingest.crewai import instrument_crewai

# Instrument CrewAI (call once at startup)
instrument_crewai()

# Choose your sink
sink = JsonlFileSink("traces/output.jsonl")  # Local file
# sink = HttpBatchSink(endpoint_url="...", api_key="...")  # Remote

# Run your crew inside an ArzuleRun context
with ArzuleRun(tenant_id="...", project_id="...", sink=sink) as run:
    result = crew.kickoff(inputs={...})
```

## What Gets Captured

The SDK automatically captures:

- **Run lifecycle** - `run.start`, `run.end`
- **Crew execution** - `crew.kickoff.start`, `crew.kickoff.complete`, `crew.kickoff.failed`
- **Agent activity** - `agent.execution.start`, `agent.execution.complete`
- **Task progress** - `task.start`, `task.complete`, `task.failed`
- **Tool calls** - `tool.call.start`, `tool.call.end`
- **LLM calls** - `llm.call.start`, `llm.call.end`
- **Handoffs** - `handoff.proposed`, `handoff.ack`, `handoff.complete`

## TraceEvent Format

Each event follows the `trace_event.v0_1` schema:

```json
{
  "schema_version": "trace_event.v0_1",
  "run_id": "uuid",
  "tenant_id": "uuid",
  "project_id": "uuid",
  "trace_id": "32hex",
  "span_id": "16hex",
  "parent_span_id": "16hex|null",
  "seq": 123,
  "ts": "2025-12-24T07:12:03.123Z",
  "agent": { "id": "crewai:role:Writer", "role": "Writer" },
  "event_type": "tool.call.start",
  "status": "ok|error|blocked|null",
  "summary": "short description",
  "attrs_compact": { "tool_name": "Search" }
}
```

## CLI

View trace files locally:

```bash
# Timeline view
arzule view traces/output.jsonl

# Table format
arzule view traces/output.jsonl -f table

# JSON output
arzule view traces/output.jsonl -f json

# Statistics
arzule stats traces/output.jsonl
```

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `ARZULE_API_KEY` | API key for authentication | Required |
| `ARZULE_TENANT_ID` | Your tenant ID | Required |
| `ARZULE_PROJECT_ID` | Your project ID | Required |
| `ARZULE_INGEST_URL` | Ingest endpoint URL | Arzule Cloud |
| `ARZULE_BATCH_SIZE` | Events per batch | 100 |
| `ARZULE_REDACT_SECRETS` | Redact secrets in payloads | true |

## PII Redaction

The SDK automatically redacts sensitive data from trace payloads:

- API keys and tokens
- Passwords and secrets
- Credit card numbers
- SSNs and other PII patterns

To disable (not recommended for production):
```python
arzule_ingest.init(redact_secrets=False)
```

## License

MIT
