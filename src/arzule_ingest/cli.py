"""CLI tools for viewing trace files locally."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def load_trace(path: str) -> list[dict[str, Any]]:
    """Load events from a JSONL trace file."""
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def format_timeline(events: list[dict[str, Any]], show_llm: bool = True) -> str:
    """Format events as a timeline view."""
    lines = []
    
    if not events:
        return "No events found."
    
    # Header
    lines.append("=" * 80)
    lines.append(f"TRACE: {events[0].get('trace_id', 'unknown')}")
    lines.append(f"RUN:   {events[0].get('run_id', 'unknown')}")
    lines.append(f"Tenant: {events[0].get('tenant_id', 'unknown')} / Project: {events[0].get('project_id', 'unknown')}")
    lines.append("=" * 80)
    lines.append("")
    
    for e in events:
        event_type = e.get("event_type", "unknown")
        
        # Skip LLM events if not requested
        if not show_llm and "llm" in event_type:
            continue
        
        ts = e.get("ts", "")[:19].replace("T", " ")
        seq = e.get("seq", 0)
        agent = e.get("agent") or {}
        agent_role = agent.get("role", "") if agent else ""
        attrs = e.get("attrs_compact") or {}
        status = e.get("status", "ok")
        
        # Indentation and icon based on event type
        indent = ""
        icon = ""
        if "run" in event_type:
            icon = "[RUN]"
        elif "crew" in event_type:
            icon = "[CREW]"
        elif "task" in event_type:
            indent = "  "
            icon = "[TASK]"
        elif "agent" in event_type:
            indent = "    "
            icon = "[AGENT]"
        elif "tool" in event_type:
            indent = "      "
            icon = "[TOOL]"
        elif "llm" in event_type:
            indent = "      "
            icon = "[LLM]"
        elif "handoff" in event_type:
            indent = "    "
            icon = "[HANDOFF]"
        else:
            icon = "[EVENT]"
        
        # Direction indicator
        direction = "   "
        if "start" in event_type:
            direction = "-->"
        elif "complete" in event_type or "end" in event_type:
            direction = "<--"
        elif "error" in event_type or "failed" in event_type:
            direction = "XXX"
        elif "ack" in event_type:
            direction = "<->"
        
        # Status indicator for errors
        status_mark = ""
        if status == "error":
            status_mark = " [ERROR]"
        
        # Build description
        desc_parts = []
        if agent_role:
            desc_parts.append(agent_role)
        
        tool_name = attrs.get("tool_name")
        if tool_name:
            desc_parts.append(f"tool={tool_name}")
        
        task_id = e.get("task_id")
        if task_id:
            desc_parts.append(f"task={task_id[:8]}...")
        
        # Event type suffix
        event_suffix = event_type.split(".")[-1]
        if desc_parts:
            desc = f"{': '.join(desc_parts)} ({event_suffix})"
        else:
            desc = event_suffix
        
        line = f"{seq:2d}. {ts} {indent}{icon:10s} {direction} {desc}{status_mark}"
        lines.append(line)
    
    lines.append("")
    lines.append("=" * 80)
    
    # Summary
    event_counts: dict[str, int] = {}
    for e in events:
        etype = e.get("event_type", "unknown").split(".")[0]
        event_counts[etype] = event_counts.get(etype, 0) + 1
    
    lines.append(f"Total events: {len(events)}")
    lines.append(f"Breakdown: {', '.join(f'{k}={v}' for k, v in sorted(event_counts.items()))}")
    
    return "\n".join(lines)


def format_table(events: list[dict[str, Any]]) -> str:
    """Format events as a compact table."""
    lines = []
    
    # Header
    lines.append(f"{'Seq':>3} {'Event Type':<25} {'Agent':<12} {'Tool':<20} {'Status':<6}")
    lines.append("-" * 70)
    
    for e in events:
        seq = e.get("seq", 0)
        event_type = e.get("event_type", "unknown")
        agent = e.get("agent") or {}
        agent_role = agent.get("role", "-") if agent else "-"
        attrs = e.get("attrs_compact") or {}
        tool_name = attrs.get("tool_name", "-")
        status = e.get("status", "ok")
        
        lines.append(f"{seq:>3} {event_type:<25} {agent_role:<12} {tool_name:<20} {status:<6}")
    
    return "\n".join(lines)


def format_json(events: list[dict[str, Any]], pretty: bool = True) -> str:
    """Format events as JSON."""
    if pretty:
        return json.dumps(events, indent=2)
    return json.dumps(events)


def format_tree(events: list[dict[str, Any]]) -> str:
    """Format events as a tree based on span hierarchy."""
    lines = []
    
    if not events:
        return "No events found."
    
    # Build span hierarchy
    span_to_event: dict[str, dict] = {}
    children: dict[str, list[str]] = {}
    roots: list[str] = []
    
    for e in events:
        span_id = e.get("span_id")
        parent_id = e.get("parent_span_id")
        
        if span_id:
            span_to_event[span_id] = e
            if parent_id:
                if parent_id not in children:
                    children[parent_id] = []
                children[parent_id].append(span_id)
            else:
                roots.append(span_id)
    
    def render_span(span_id: str, depth: int = 0) -> None:
        e = span_to_event.get(span_id)
        if not e:
            return
        
        event_type = e.get("event_type", "unknown")
        agent = e.get("agent") or {}
        agent_role = agent.get("role", "") if agent else ""
        attrs = e.get("attrs_compact") or {}
        tool_name = attrs.get("tool_name", "")
        
        prefix = "  " * depth
        
        desc = event_type
        if agent_role:
            desc += f" [{agent_role}]"
        if tool_name:
            desc += f" ({tool_name})"
        
        lines.append(f"{prefix}- {desc}")
        
        for child_id in children.get(span_id, []):
            render_span(child_id, depth + 1)
    
    lines.append("Span Tree:")
    lines.append("-" * 40)
    for root_id in roots:
        render_span(root_id)
    
    return "\n".join(lines)


def cmd_view(args: argparse.Namespace) -> int:
    """View a trace file."""
    try:
        events = load_trace(args.file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in trace file: {e}", file=sys.stderr)
        return 1
    
    if args.format == "timeline":
        output = format_timeline(events, show_llm=not args.no_llm)
    elif args.format == "table":
        output = format_table(events)
    elif args.format == "json":
        output = format_json(events, pretty=not args.compact)
    elif args.format == "tree":
        output = format_tree(events)
    else:
        output = format_timeline(events, show_llm=not args.no_llm)
    
    print(output)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show statistics for a trace file."""
    try:
        events = load_trace(args.file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        return 1
    
    if not events:
        print("No events found.")
        return 0
    
    # Basic stats
    print(f"Trace ID: {events[0].get('trace_id', 'unknown')}")
    print(f"Run ID: {events[0].get('run_id', 'unknown')}")
    print(f"Total events: {len(events)}")
    print()
    
    # Time range
    timestamps = [e.get("ts", "") for e in events if e.get("ts")]
    if timestamps:
        print(f"Start: {timestamps[0]}")
        print(f"End: {timestamps[-1]}")
    print()
    
    # Event type breakdown
    event_counts: dict[str, int] = {}
    for e in events:
        etype = e.get("event_type", "unknown")
        event_counts[etype] = event_counts.get(etype, 0) + 1
    
    print("Event counts:")
    for etype, count in sorted(event_counts.items()):
        print(f"  {etype}: {count}")
    print()
    
    # Agent breakdown
    agents: set[str] = set()
    for e in events:
        agent = e.get("agent") or {}
        role = agent.get("role")
        if role:
            agents.add(role)
    
    if agents:
        print(f"Agents: {', '.join(sorted(agents))}")
    
    # Error count
    errors = [e for e in events if e.get("status") == "error"]
    if errors:
        print(f"Errors: {len(errors)}")
        for err in errors:
            print(f"  - {err.get('event_type')}: {err.get('summary', '')[:50]}")
    
    return 0


def main() -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="arzule",
        description="Arzule trace viewer - inspect JSONL trace files locally",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # view command
    view_parser = subparsers.add_parser("view", help="View a trace file")
    view_parser.add_argument("file", help="Path to the JSONL trace file")
    view_parser.add_argument(
        "-f", "--format",
        choices=["timeline", "table", "json", "tree"],
        default="timeline",
        help="Output format (default: timeline)",
    )
    view_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Hide LLM call events",
    )
    view_parser.add_argument(
        "--compact",
        action="store_true",
        help="Compact JSON output (no indentation)",
    )
    view_parser.set_defaults(func=cmd_view)

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show trace statistics")
    stats_parser.add_argument("file", help="Path to the JSONL trace file")
    stats_parser.set_defaults(func=cmd_stats)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

