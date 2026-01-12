# Technology Stack

**Analysis Date:** 2026-01-10

## Languages

**Primary:**
- Python 3.10+ - All application code (`pyproject.toml`)

**Secondary:**
- None - Pure Python project

## Runtime

**Environment:**
- Python 3.10-3.14 supported (`pyproject.toml`)
- No browser runtime (SDK/CLI tool)

**Package Manager:**
- pip with pyproject.toml (PEP 517/518)
- Build backend: Hatchling
- Lockfile: None (uses version constraints in pyproject.toml)

## Frameworks

**Core:**
- None (vanilla Python SDK)

**Multi-Agent Framework Integrations (all optional):**
- CrewAI >= 0.80.0 - `src/arzule_ingest/crewai/`
- LangChain >= 0.2.0 - `src/arzule_ingest/langchain/`
- LangGraph (auto-detected) - `src/arzule_ingest/langgraph/`
- AutoGen v0.2 (pyautogen) - `src/arzule_ingest/autogen/`
- AutoGen v0.7+ (autogen-core, autogen-agentchat) - `src/arzule_ingest/autogen_v2/`
- Claude Code - `src/arzule_ingest/claude/`

**Testing:**
- pytest >= 8.0.0 - Unit and integration tests
- pytest-asyncio >= 0.23.0 - Async test support

**Build/Dev:**
- Hatchling - Package building
- langchain-openai >= 0.1.0 - Dev dependency for LangChain tests

## Key Dependencies

**Critical:**
- httpx >= 0.27.0 - HTTP client for API communication (`src/arzule_ingest/sinks/http_batch.py`)
- pydantic >= 2.0.0 - Data validation (`src/arzule_ingest/config.py`)
- cryptography >= 42.0.0 - AES-256-Fernet encryption for SOC2 compliance (`src/arzule_ingest/audit.py`)

**Infrastructure:**
- Python stdlib - threading, contextvars, json, secrets, uuid
- No database dependencies (stateless SDK)

## Configuration

**Environment:**
- Environment variables: `ARZULE_API_KEY`, `ARZULE_TENANT_ID`, `ARZULE_PROJECT_ID`, `ARZULE_INGEST_URL`
- User config file: `~/.arzule/config` (key=value format)
- Project .env file for development (gitignored)

**Build:**
- `pyproject.toml` - Project metadata, dependencies, build config
- No separate TypeScript or other config files

## Platform Requirements

**Development:**
- macOS/Linux/Windows (any platform with Python 3.10+)
- No external tooling dependencies
- Virtual environment recommended (`.venv/`)

**Production:**
- Distributed as PyPI package: `arzule-ingest`
- Installed via: `pip install arzule-ingest`
- Runs on user's Python installation

**CLI Commands:**
- `arzule` - Main CLI (`src/arzule_ingest/cli.py`)
- `arzule-claude` - Claude Code wrapper (`src/arzule_ingest/claude/wrapper.py`)
- `arzule-claude-install` - Hook installation (`src/arzule_ingest/claude/install.py`)

---

*Stack analysis: 2026-01-10*
*Update after major dependency changes*
