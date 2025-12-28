# arzule-ingest

Lightweight SDK for capturing multi-agent traces and sending them to Arzule.

**Supported Frameworks:**
- CrewAI
- LangChain / LangGraph
- Microsoft AutoGen

## Installation

```bash
pip install arzule-ingest
```

## Quick Start

### Option 1: One-line setup (recommended)

```python
import arzule_ingest

# Initialize with environment variables
# Auto-detects and instruments CrewAI, LangChain, and AutoGen if installed
arzule_ingest.init()

# Your agent code runs as normal - traces are captured automatically
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
```

## Framework-Specific Usage

### CrewAI

```python
from arzule_ingest import ArzuleRun
from arzule_ingest.sinks import JsonlFileSink
from arzule_ingest.crewai import instrument_crewai

# Instrument CrewAI (call once at startup)
instrument_crewai()

# Run your crew inside an ArzuleRun context
sink = JsonlFileSink("traces/output.jsonl")
with ArzuleRun(tenant_id="...", project_id="...", sink=sink) as run:
    result = crew.kickoff(inputs={...})
```

### LangChain / LangGraph

```python
from arzule_ingest import ArzuleRun
from arzule_ingest.sinks import JsonlFileSink
from arzule_ingest.langchain import instrument_langchain

# Instrument LangChain and get the callback handler
handler = instrument_langchain()

# Use the handler with your chains/agents
sink = JsonlFileSink("traces/output.jsonl")
with ArzuleRun(tenant_id="...", project_id="...", sink=sink) as run:
    # Pass handler to invoke()
    result = chain.invoke({"input": "..."}, config={"callbacks": [handler]})
    
    # Or use with agents
    result = agent.invoke({"input": "..."}, config={"callbacks": [handler]})
```

### Microsoft AutoGen

```python
from arzule_ingest import ArzuleRun
from arzule_ingest.sinks import JsonlFileSink
from arzule_ingest.autogen import instrument_autogen
from autogen import AssistantAgent, UserProxyAgent

# Instrument AutoGen (call once at startup)
instrument_autogen()

# Create your agents
assistant = AssistantAgent("assistant", llm_config={...})
user_proxy = UserProxyAgent("user_proxy", ...)

# Run inside an ArzuleRun context
sink = JsonlFileSink("traces/output.jsonl")
with ArzuleRun(tenant_id="...", project_id="...", sink=sink) as run:
    user_proxy.initiate_chat(assistant, message="Hello!")
```

## What Gets Captured

The SDK automatically captures framework-specific events:

### All Frameworks
- **Run lifecycle** - `run.start`, `run.end`
- **LLM calls** - `llm.call.start`, `llm.call.end`
- **Tool calls** - `tool.call.start`, `tool.call.end`

### CrewAI
- **Crew execution** - `crew.kickoff.start`, `crew.kickoff.complete`
- **Agent activity** - `agent.execution.start`, `agent.execution.complete`
- **Task progress** - `task.start`, `task.complete`, `task.failed`
- **Handoffs** - `handoff.proposed`, `handoff.ack`, `handoff.complete`

### LangChain
- **Chain execution** - `chain.start`, `chain.end`, `chain.error`
- **Agent actions** - `agent.action`, `agent.finish`
- **Retriever calls** - `retriever.start`, `retriever.end`

### AutoGen
- **Messages** - `agent.message.send`, `agent.message.receive`
- **Conversations** - `conversation.start`, `conversation.end`
- **Code execution** - `code.execution`
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
  "workstream_id": null,
  "task_id": null,
  "event_type": "tool.call.start",
  "status": "ok|error|blocked|null",
  "summary": "short description",
  "attrs_compact": { "tool_name": "Search" },
  "payload": {},
  "raw_ref": { "storage": "inline" }
}
```

## Instrumentation Modes

All integrations support two modes:

```python
# Full instrumentation (default)
instrument_crewai(mode="global")
instrument_langchain(mode="global")
instrument_autogen(mode="global")

# Minimal instrumentation (lifecycle events only)
instrument_crewai(mode="minimal")
instrument_langchain(mode="minimal")
instrument_autogen(mode="minimal")
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
| `ARZULE_REDACT_PII` | Redact PII in payloads | true |

## PII Redaction

The SDK automatically redacts sensitive data from trace payloads:

- API keys and tokens
- Passwords and secrets
- Email addresses
- Phone numbers
- Credit card numbers
- SSNs and other PII patterns

To disable (not recommended for production):
```bash
export ARZULE_REDACT_PII=false
```

## License

MIT
