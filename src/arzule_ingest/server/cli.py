"""CLI for running the self-hosted Arzule backend.

Usage:
    arzule-server --port 8080 --storage ./traces
    arzule-server --host 0.0.0.0 --port 8080

Or via Python:
    python -m arzule_ingest.server.cli --port 8080
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    """Run the self-hosted Arzule backend server."""
    parser = argparse.ArgumentParser(
        description="Self-hosted Arzule backend for trace ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start server on default port
    arzule-server

    # Start server on custom port with custom storage
    arzule-server --port 8080 --storage /data/traces

    # Bind to all interfaces (for Docker/k8s)
    arzule-server --host 0.0.0.0 --port 8080

Environment variables:
    ARZULE_STORAGE_DIR  - Storage directory (default: ./traces)
        """,
    )

    parser.add_argument(
        "--host",
        "-H",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--storage",
        "-s",
        default="./traces",
        help="Directory to store trace files (default: ./traces)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )

    args = parser.parse_args()

    # Check for uvicorn
    try:
        import uvicorn
    except ImportError:
        print(
            "Error: uvicorn is required for the self-hosted server.\n"
            "Install with: pip install arzule-ingest[server]",
            file=sys.stderr,
        )
        sys.exit(1)

    # Set storage directory
    os.environ["ARZULE_STORAGE_DIR"] = args.storage

    print(f"Starting Arzule self-hosted backend...")
    print(f"  Storage: {args.storage}")
    print(f"  Endpoint: http://{args.host}:{args.port}/ingest")
    print()
    print("To connect your SDK:")
    print(f'  export ARZULE_MODE=selfhosted')
    print(f'  export ARZULE_SELFHOSTED_ENDPOINT=http://{args.host}:{args.port}/ingest')
    print()

    # Run the server
    uvicorn.run(
        "arzule_ingest.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
