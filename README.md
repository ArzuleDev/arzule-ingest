# arzule-ingest

Lightweight SDK for capturing multi-agent traces and sending them to Arzule.

**Supported Frameworks:**
- CrewAI
- LangChain / LangGraph
- Microsoft AutoGen (both legacy v0.2 and new v0.7+)
- Claude Code (via hooks + OTel)

## Installation

The SDK has a lightweight core with optional framework integrations. Install only what you need:

```bash
# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Core only (httpx, pydantic, cryptography)
pip install arzule-ingest

# With specific framework support
pip install arzule-ingest[crewai]      # CrewAI integration
pip install arzule-ingest[langchain]   # LangChain/LangGraph integration
pip install arzule-ingest[autogen]     # AutoGen v0.2 integration

# Multiple frameworks
pip install arzule-ingest[crewai,langchain]

# All framework integrations
pip install arzule-ingest[all]
```

> **Note:** Framework integrations require Python versions supported by each framework. For example, CrewAI currently requires Python <3.14. The core SDK works on Python 3.10+.

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
import os
import arzule_ingest

# Always use environment variables for credentials - NEVER hardcode API keys
arzule_ingest.init(
    api_key=os.environ["ARZULE_API_KEY"],
    tenant_id=os.environ["ARZULE_TENANT_ID"],
    project_id=os.environ["ARZULE_PROJECT_ID"],
)
```

> **Security Note:** Never hardcode API keys in source code. Use environment variables or a secrets manager.

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

### Microsoft AutoGen (Legacy v0.2)

```python
from arzule_ingest import ArzuleRun
from arzule_ingest.sinks import JsonlFileSink
from arzule_ingest.autogen import instrument_autogen
from autogen import AssistantAgent, UserProxyAgent

# Instrument AutoGen v0.2 (call once at startup)
instrument_autogen()

# Create your agents
assistant = AssistantAgent("assistant", llm_config={...})
user_proxy = UserProxyAgent("user_proxy", ...)

# Run inside an ArzuleRun context
sink = JsonlFileSink("traces/output.jsonl")
with ArzuleRun(tenant_id="...", project_id="...", sink=sink) as run:
    user_proxy.initiate_chat(assistant, message="Hello!")
```

### Microsoft AutoGen v0.7+ (New Architecture)

```python
import asyncio
import arzule_ingest

# Initialize (automatically detects and instruments v0.7+)
arzule_ingest.init()

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    agent = AssistantAgent("assistant", model_client=model_client)
    
    # Run task - automatically traced
    result = await agent.run(task="What is 2 + 2?")
    print(result.messages[-1].to_text())
    
    await model_client.close()

asyncio.run(main())
```

**Note:** The SDK automatically detects which version of AutoGen you have installed. For detailed AutoGen v0.7+ documentation, see [AUTOGEN_V2_INTEGRATION.md](AUTOGEN_V2_INTEGRATION.md).

### Claude Code

Capture traces from Claude Code CLI sessions with full observability.

#### Quick Start (Recommended)

Use the `arzule-claude` wrapper for complete observability (hooks + OTel metrics):

```bash
# Instead of running:
$ claude "your prompt"

# Run:
$ arzule-claude "your prompt"

# All arguments pass through:
$ arzule-claude --model opus "your prompt"
```

This captures:
- **Hooks data**: Session events, tool calls, user prompts, subagent activity
- **OTel metrics**: Token usage, costs, latency, cache metrics

#### Installation

```bash
# Install the SDK
pip install arzule-ingest

# Configure credentials (one-time setup)
arzule configure
```

#### Alternative: Hooks-Only Installation

If you prefer to use the regular `claude` command with just hooks (no OTel metrics):

```bash
# Install hooks into Claude Code settings
arzule-claude-install install

# Check installation status
arzule-claude-install status

# Uninstall hooks
arzule-claude-install uninstall
```

#### What Gets Captured

| Data Type | `arzule-claude` | Hooks-only |
|-----------|-----------------|------------|
| Session start/end | ✅ | ✅ |
| User prompts | ✅ | ✅ |
| Tool calls (pre/post) | ✅ | ✅ |
| Subagent activity | ✅ | ✅ |
| Token usage | ✅ | ❌ |
| Costs | ✅ | ❌ |
| Latency metrics | ✅ | ❌ |

#### Configuration

The wrapper reads configuration from `~/.arzule/config` (created by `arzule configure`) or environment variables:

```bash
export ARZULE_API_KEY="your-api-key"
export ARZULE_TENANT_ID="your-tenant-id"
export ARZULE_PROJECT_ID="your-project-id"
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

### AutoGen (v0.2 and v0.7+)
- **Messages** - `agent.message.send`, `agent.message.receive`
- **Agent lifecycle** - `agent.start`, `agent.end`
- **Conversations** - `conversation.start`, `conversation.end`
- **Code execution** - `code.execution` (v0.2 only)
- **Handoffs** - `handoff.proposed`, `handoff.ack`, `handoff.complete`
- **Agent events** - `agent.event` (v0.7+ only)

### Claude Code
- **Sessions** - `session.start`, `session.end`
- **Turns** - `turn.start`, `turn.end`
- **Tool calls** - `tool.call.start`, `tool.call.end`
- **Subagents** - `subagent.start`, `subagent.stop`
- **User prompts** - `user.prompt.submit`
- **Notifications** - `notification`

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
