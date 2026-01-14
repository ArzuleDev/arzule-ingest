"""Minimal self-hosted backend for Arzule traces.

This provides a simple HTTP server that accepts JSONL trace events
and stores them in local files organized by run_id.

Endpoints:
    POST /ingest - Accept JSONL trace events
    GET /health - Health check
    GET /runs - List stored runs
    GET /runs/{run_id} - Get events for a specific run
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    from fastapi import FastAPI, Request, HTTPException, Response
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    raise ImportError(
        "FastAPI is required for the self-hosted server. "
        "Install with: pip install arzule-ingest[server]"
    )


def create_app(
    storage_dir: str = "./traces",
    enable_cors: bool = True,
    allowed_origins: Optional[list[str]] = None,
) -> FastAPI:
    """
    Create the FastAPI application for self-hosted Arzule backend.

    Args:
        storage_dir: Directory to store trace files (default: ./traces)
        enable_cors: Enable CORS for browser-based clients
        allowed_origins: List of allowed CORS origins (default: ["*"])

    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="Arzule Self-Hosted Backend",
        description="Minimal trace ingestion server for Arzule SDK",
        version="0.9.0",
    )

    # Configure storage
    storage_path = Path(storage_dir)
    storage_path.mkdir(parents=True, exist_ok=True)

    # Enable CORS if requested
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.post("/ingest")
    async def ingest(request: Request) -> dict[str, Any]:
        """
        Accept JSONL trace events.

        Request body should be newline-delimited JSON (JSONL format).
        Each line is a trace event with at least a run_id field.

        Events are appended to run-specific files: {storage_dir}/{run_id}.jsonl
        """
        try:
            content = await request.body()
            lines = content.decode("utf-8").strip().split("\n")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

        events = []
        errors = []

        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                events.append(event)
            except json.JSONDecodeError as e:
                errors.append({"line": i + 1, "error": str(e)})

        if not events:
            if errors:
                raise HTTPException(status_code=400, detail={"errors": errors})
            return {"status": "ok", "events": 0}

        # Group events by run_id
        events_by_run: dict[str, list[dict]] = {}
        for event in events:
            run_id = event.get("run_id", "unknown")
            if run_id not in events_by_run:
                events_by_run[run_id] = []
            events_by_run[run_id].append(event)

        # Append to run-specific files
        for run_id, run_events in events_by_run.items():
            trace_file = storage_path / f"{run_id}.jsonl"
            with open(trace_file, "a") as f:
                for event in run_events:
                    f.write(json.dumps(event, separators=(",", ":"), default=str) + "\n")

        return {
            "status": "ok",
            "events": len(events),
            "runs": list(events_by_run.keys()),
            "errors": errors if errors else None,
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "storage_dir": str(storage_path.absolute()),
        }

    @app.get("/runs")
    async def list_runs(
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List all stored runs.

        Returns run_id, file path, size, and modification time.
        Sorted by modification time (newest first).
        """
        runs = []
        for f in storage_path.glob("*.jsonl"):
            stat = f.stat()
            runs.append({
                "run_id": f.stem,
                "path": str(f),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "event_count": sum(1 for _ in open(f)),
            })

        # Sort by modification time (newest first)
        runs.sort(key=lambda r: r["modified_at"], reverse=True)

        # Apply pagination
        total = len(runs)
        runs = runs[offset : offset + limit]

        return {
            "runs": runs,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        """
        Get all events for a specific run.

        Returns events in chronological order (by seq if available).
        """
        trace_file = storage_path / f"{run_id}.jsonl"
        if not trace_file.exists():
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

        events = []
        with open(trace_file) as f:
            for line in f:
                if line.strip():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # Sort by seq if available
        events.sort(key=lambda e: e.get("seq", 0))

        return {
            "run_id": run_id,
            "events": events,
            "event_count": len(events),
        }

    @app.delete("/runs/{run_id}")
    async def delete_run(run_id: str) -> dict[str, Any]:
        """Delete a specific run."""
        trace_file = storage_path / f"{run_id}.jsonl"
        if not trace_file.exists():
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

        trace_file.unlink()
        return {"status": "deleted", "run_id": run_id}

    return app


# Default app instance for uvicorn
app = create_app(
    storage_dir=os.environ.get("ARZULE_STORAGE_DIR", "./traces")
)
