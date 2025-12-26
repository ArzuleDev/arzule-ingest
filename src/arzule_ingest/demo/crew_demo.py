"""Demo crew with delegation to test handoff tracking."""

from __future__ import annotations

import os
from pathlib import Path


def _load_env() -> None:
    """Load environment variables from .env file if it exists."""
    env_path = Path(__file__).parents[3] / ".env"
    if not env_path.exists():
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and value and key not in os.environ:
                os.environ[key] = value


def run_demo(output_path: str = "out/demo_trace.jsonl") -> None:
    """
    Run a demo crew with delegation to generate trace events.

    This creates a simple crew with two agents where one delegates to the other,
    allowing you to verify handoff tracking works correctly.

    Args:
        output_path: Path to write the JSONL trace file
    """
    # Load .env file first
    _load_env()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set.")
        print("Create a .env file in the project root with:")
        print("  ANTHROPIC_API_KEY=sk-ant-...")
        return

    # Import here to avoid requiring crewai for non-demo usage
    try:
        from crewai import Agent, Crew, LLM, Task
    except ImportError:
        print("CrewAI not installed. Run: pip install 'crewai[tools]'")
        return

    from ..crewai import instrument_crewai
    from ..run import ArzuleRun
    from ..sinks.file_jsonl import JsonlFileSink

    # Instrument CrewAI before creating any crews
    instrument_crewai()

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Configure Anthropic LLM
    llm = LLM(model="anthropic/claude-3-haiku-20240307")

    # Create agents
    researcher = Agent(
        role="Researcher",
        goal="Find accurate information on topics",
        backstory="You are an expert researcher who can find and synthesize information.",
        llm=llm,
        verbose=True,
        allow_delegation=True,  # Can delegate to writer
    )

    writer = Agent(
        role="Writer",
        goal="Write clear and engaging content",
        backstory="You are a skilled writer who creates compelling content.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # Create tasks that trigger delegation
    research_task = Task(
        description="Research the topic of 'observability in AI systems' and compile key findings. "
        "If you need help writing a summary, delegate to the Writer.",
        expected_output="A comprehensive research summary",
        agent=researcher,
    )

    writing_task = Task(
        description="Write a brief article based on the research findings.",
        expected_output="A well-written article",
        agent=writer,
    )

    # Create crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        verbose=True,
    )

    # Run with Arzule tracing
    sink = JsonlFileSink(output_path)

    tenant_id = os.environ.get("ARZULE_TENANT_ID", "demo-tenant")
    project_id = os.environ.get("ARZULE_PROJECT_ID", "demo-project")

    print(f"Running demo crew with tracing to: {output_path}")

    with ArzuleRun(tenant_id=tenant_id, project_id=project_id, sink=sink) as run:
        print(f"Run ID: {run.run_id}")
        print(f"Trace ID: {run.trace_id}")

        try:
            result = crew.kickoff()
            print(f"\nCrew result: {result}")
        except Exception as e:
            print(f"\nCrew failed: {e}")

    print(f"\nTrace written to: {output_path}")
    print("View with: cat {output_path} | jq .")


if __name__ == "__main__":
    run_demo()

