"""Tests for handoff key injection and extraction."""

from arzule_ingest.crewai.handoff import (
    DELEGATION_TOOL_NAMES,
    extract_handoff_key_from_text,
    is_delegation_tool,
    maybe_inject_handoff_key,
)
from arzule_ingest.run import ArzuleRun
from arzule_ingest.sinks.base import TelemetrySink


class MockSink(TelemetrySink):
    """Mock sink for testing."""

    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(event)

    def flush(self):
        pass


class MockContext:
    """Mock tool context for testing."""

    def __init__(self, tool_name: str, tool_input: dict, agent=None):
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.agent = agent


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, role: str):
        self.role = role


class TestIsDelegationTool:
    """Tests for is_delegation_tool function."""

    def test_exact_match(self):
        """Test exact match of known delegation tools."""
        for name in DELEGATION_TOOL_NAMES:
            assert is_delegation_tool(name) is True

    def test_coworker_keyword(self):
        """Test that tools with 'coworker' are detected."""
        assert is_delegation_tool("hand_off_to_coworker") is True
        assert is_delegation_tool("COWORKER_tool") is True
        assert is_delegation_tool("my_coworker_helper") is True

    def test_non_delegation_tools(self):
        """Test that normal tools are not flagged."""
        assert is_delegation_tool("search") is False
        assert is_delegation_tool("calculate") is False
        assert is_delegation_tool("read_file") is False

    def test_none_and_empty(self):
        """Test handling of None and empty strings."""
        assert is_delegation_tool(None) is False
        assert is_delegation_tool("") is False


class TestMaybeInjectHandoffKey:
    """Tests for maybe_inject_handoff_key function."""

    def test_injects_key_for_delegation(self):
        """Test that handoff key is injected for delegation tools."""
        sink = MockSink()
        run = ArzuleRun(tenant_id="t1", project_id="p1", sink=sink)
        agent = MockAgent("Researcher")

        context = MockContext(
            tool_name="delegate_work_to_coworker",
            tool_input={"task": "Write a summary", "coworker": "Writer"},
            agent=agent,
        )

        key = maybe_inject_handoff_key(run, context)

        assert key is not None
        assert len(key) == 36  # UUID format
        assert "arzule" in context.tool_input
        assert context.tool_input["arzule"]["handoff_key"] == key
        assert context.tool_input["task"].startswith(f"[arzule_handoff:{key}]")

    def test_skips_non_delegation_tools(self):
        """Test that normal tools are not modified."""
        sink = MockSink()
        run = ArzuleRun(tenant_id="t1", project_id="p1", sink=sink)

        context = MockContext(
            tool_name="search",
            tool_input={"query": "test"},
        )

        key = maybe_inject_handoff_key(run, context)

        assert key is None
        assert "arzule" not in context.tool_input

    def test_stores_pending_handoff(self):
        """Test that pending handoff metadata is stored."""
        sink = MockSink()
        run = ArzuleRun(tenant_id="t1", project_id="p1", sink=sink)
        agent = MockAgent("Manager")

        context = MockContext(
            tool_name="ask_question_to_coworker",
            tool_input={"question": "What is the status?"},
            agent=agent,
        )

        key = maybe_inject_handoff_key(run, context)

        assert key in run._handoff_pending
        assert run._handoff_pending[key]["from_role"] == "Manager"


class TestExtractHandoffKey:
    """Tests for extract_handoff_key_from_text function."""

    def test_extracts_key_from_text(self):
        """Test extraction of handoff key from text."""
        key = "12345678-1234-1234-1234-123456789abc"
        text = f"[arzule_handoff:{key}] This is the task description"

        extracted = extract_handoff_key_from_text(text)

        assert extracted == key

    def test_extracts_key_from_middle_of_text(self):
        """Test extraction when marker is in middle of text."""
        key = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        text = f"Some prefix text [arzule_handoff:{key}] and more text"

        extracted = extract_handoff_key_from_text(text)

        assert extracted == key

    def test_returns_none_for_no_marker(self):
        """Test that None is returned when no marker present."""
        text = "This is normal text without a handoff marker"

        extracted = extract_handoff_key_from_text(text)

        assert extracted is None

    def test_returns_none_for_none_input(self):
        """Test that None input returns None."""
        assert extract_handoff_key_from_text(None) is None

    def test_returns_none_for_empty_string(self):
        """Test that empty string returns None."""
        assert extract_handoff_key_from_text("") is None

