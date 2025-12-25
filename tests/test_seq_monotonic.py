"""Tests for sequence number monotonicity."""

import json

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


class TestSeqMonotonic:
    """Tests for sequence number monotonicity."""

    def test_seq_starts_at_1(self):
        """Test that sequence numbers start at 1."""
        sink = MockSink()

        with ArzuleRun(tenant_id="t1", project_id="p1", sink=sink) as run:
            pass

        # First event should have seq=1
        assert sink.events[0]["seq"] == 1

    def test_seq_increases_monotonically(self):
        """Test that sequence numbers always increase."""
        sink = MockSink()

        with ArzuleRun(tenant_id="t1", project_id="p1", sink=sink) as run:
            # Emit some events
            for i in range(5):
                run.emit(run._make_event(
                    event_type="test.event",
                    summary=f"Event {i}",
                ))

        # Check monotonicity
        seqs = [e["seq"] for e in sink.events]
        for i in range(1, len(seqs)):
            assert seqs[i] > seqs[i - 1], f"Seq {seqs[i]} not greater than {seqs[i - 1]}"

    def test_seq_has_no_gaps(self):
        """Test that sequence numbers have no gaps."""
        sink = MockSink()

        with ArzuleRun(tenant_id="t1", project_id="p1", sink=sink) as run:
            for i in range(10):
                run.emit(run._make_event(
                    event_type="test.event",
                    summary=f"Event {i}",
                ))

        seqs = [e["seq"] for e in sink.events]
        expected = list(range(1, len(seqs) + 1))
        assert seqs == expected

    def test_run_start_and_end_have_seqs(self):
        """Test that run.start and run.end events have proper seqs."""
        sink = MockSink()

        with ArzuleRun(tenant_id="t1", project_id="p1", sink=sink) as run:
            pass

        # Should have at least start and end
        assert len(sink.events) >= 2

        start_event = next(e for e in sink.events if e["event_type"] == "run.start")
        end_event = next(e for e in sink.events if e["event_type"] == "run.end")

        assert start_event["seq"] < end_event["seq"]


class TestRunIds:
    """Tests for run ID and trace ID generation."""

    def test_run_has_unique_ids(self):
        """Test that each run gets unique IDs."""
        sink = MockSink()

        run_ids = []
        trace_ids = []

        for _ in range(3):
            with ArzuleRun(tenant_id="t1", project_id="p1", sink=sink) as run:
                run_ids.append(run.run_id)
                trace_ids.append(run.trace_id)

        assert len(set(run_ids)) == 3, "Run IDs should be unique"
        assert len(set(trace_ids)) == 3, "Trace IDs should be unique"

    def test_all_events_have_same_run_id(self):
        """Test that all events in a run share the same run_id."""
        sink = MockSink()

        with ArzuleRun(tenant_id="t1", project_id="p1", sink=sink) as run:
            for i in range(5):
                run.emit(run._make_event(event_type="test.event"))

        run_ids = set(e["run_id"] for e in sink.events)
        assert len(run_ids) == 1

    def test_all_events_have_same_trace_id(self):
        """Test that all events in a run share the same trace_id."""
        sink = MockSink()

        with ArzuleRun(tenant_id="t1", project_id="p1", sink=sink) as run:
            for i in range(5):
                run.emit(run._make_event(event_type="test.event"))

        trace_ids = set(e["trace_id"] for e in sink.events)
        assert len(trace_ids) == 1


class TestEventSchema:
    """Tests for TraceEvent schema compliance."""

    def test_event_has_required_fields(self):
        """Test that events have all required fields."""
        sink = MockSink()

        required_fields = {
            "schema_version",
            "run_id",
            "tenant_id",
            "project_id",
            "trace_id",
            "span_id",
            "seq",
            "ts",
            "event_type",
            "status",
            "summary",
            "attrs_compact",
            "payload",
            "raw_ref",
        }

        with ArzuleRun(tenant_id="t1", project_id="p1", sink=sink) as run:
            run.emit(run._make_event(event_type="test.event", summary="Test"))

        for event in sink.events:
            missing = required_fields - set(event.keys())
            assert not missing, f"Missing fields: {missing}"

    def test_event_is_json_serializable(self):
        """Test that events can be serialized to JSON."""
        sink = MockSink()

        with ArzuleRun(tenant_id="t1", project_id="p1", sink=sink) as run:
            run.emit(run._make_event(
                event_type="test.event",
                attrs_compact={"key": "value", "number": 123},
                payload={"data": [1, 2, 3]},
            ))

        for event in sink.events:
            # Should not raise
            json_str = json.dumps(event)
            parsed = json.loads(json_str)
            assert parsed == event

