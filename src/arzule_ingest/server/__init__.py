"""Optional self-hosted backend for Arzule traces.

This package provides a minimal self-hosted backend that accepts JSONL trace events
from the Arzule SDK and stores them locally. It's useful for:

- Local development and testing
- Self-hosted deployments without Arzule cloud
- Building custom observability pipelines

Usage:
    pip install arzule-ingest[server]
    arzule-server --port 8080 --storage ./traces

Or programmatically:
    from arzule_ingest.server import create_app
    app = create_app(storage_dir="./traces")
"""

from .app import create_app

__all__ = ["create_app"]
