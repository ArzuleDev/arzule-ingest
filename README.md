# Arzule Ingestion Wrapper

Observability wrapper for CrewAI that emits TraceEvent JSONL with deterministic handoff edge derivation.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from arzule_ingest import ArzuleRun
from arzule_ingest.sinks import JsonlFileSink
from arzule_ingest.crewai import instrument_crewai

# 1. Instrument CrewAI (call once at startup, before creating crews)
instrument_crewai()

# 2. Create a sink for trace events
sink = JsonlFileSink("out/trace.jsonl")

# 3. Run your crew inside an ArzuleRun context
with ArzuleRun(tenant_id="your-tenant", project_id="your-project", sink=sink) as run:
    result = crew.kickoff(inputs={...})
```

## TraceEvent Format (v0.1)

Each line in the JSONL output is a TraceEvent:

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
  "task_id": "optional",
  "event_type": "tool.call.start",
  "status": "ok|error|blocked|null",
  "summary": "short string",
  "attrs_compact": { "tool_name": "Search", "handoff_key": "..." },
  "payload": { "tool_input": {...} },
  "raw_ref": { "storage": "inline" }
}
```

## Event Types

- `run.start` / `run.end` - Run lifecycle
- `crew.kickoff.start` / `crew.kickoff.complete` / `crew.kickoff.failed`
- `agent.execution.start` / `agent.execution.complete` / `agent.execution.failed`
- `task.start` / `task.complete` / `task.failed`
- `tool.call.start` / `tool.call.end`
- `llm.call.start` / `llm.call.end`
- `handoff.proposed` / `handoff.ack` / `handoff.complete`

## Handoff Tracking

Delegation calls (e.g., `delegate_work_to_coworker`) are automatically detected. The wrapper:

1. Injects a `handoff_key` UUID into the tool input
2. Emits `handoff.proposed` when delegation starts
3. Emits `handoff.ack` when the receiving agent picks up the task
4. Emits `handoff.complete` when the delegated task finishes

This enables deterministic `handoff_edges` materialization in your backend.

## Configuration

Environment variables:

- `ARZULE_API_KEY` - API key for HTTP sink
- `ARZULE_INGEST_URL` - Ingest endpoint URL
- `ARZULE_TENANT_ID` - Default tenant ID
- `ARZULE_PROJECT_ID` - Default project ID
- `ARZULE_BATCH_SIZE` - Events per batch (default: 100)
- `ARZULE_REDACT_SECRETS` - Redact secrets in payloads (default: true)
- `CREWAI_DISABLE_TELEMETRY` - Disable CrewAI's own telemetry

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Demo

```bash
python -m arzule_ingest.demo.crew_demo
```
