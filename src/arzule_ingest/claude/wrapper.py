"""Wrapper to launch Claude Code with OTel configured for Arzule.

Instead of running `claude` directly, users run `arzule-claude` which:
1. Sets up OTel environment variables to export to Arzule (Claude Code appends /v1/traces)
2. Launches the real `claude` CLI with all args passed through

This gives you both:
- Hooks-based qualitative data (transcripts, tool payloads, HITL)
- OTel-based quantitative data (tokens, costs, latency, cache metrics)

Usage:
    # Instead of:
    $ claude "your prompt"
    
    # Run:
    $ arzule-claude "your prompt"
    
    # Or with arguments:
    $ arzule-claude --model opus "your prompt"
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ..endpoints import get_ingest_base_url


def _load_arzule_config() -> dict[str, str]:
    """
    Load Arzule configuration from ~/.arzule/config.
    
    Returns dict of environment variables to set.
    """
    config = {}
    config_path = Path.home() / ".arzule" / "config"
    
    if not config_path.exists():
        return config
    
    try:
        for line in config_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                config[key] = value
    except Exception:
        pass
    
    return config


def _get_otel_endpoint(config: dict[str, str]) -> str:
    """
    Determine the OTel endpoint URL.
    
    Priority:
    1. ARZULE_OTEL_ENDPOINT env var (explicit override)
    2. ARZULE_INGEST_URL base URL (Claude Code appends /v1/traces)
    3. Default production endpoint
    """
    # Explicit OTel endpoint override
    if os.environ.get("ARZULE_OTEL_ENDPOINT"):
        return os.environ["ARZULE_OTEL_ENDPOINT"]
    
    if config.get("ARZULE_OTEL_ENDPOINT"):
        return config["ARZULE_OTEL_ENDPOINT"]
    
    # Derive from ingest URL
    ingest_url = (
        os.environ.get("ARZULE_INGEST_URL") or
        config.get("ARZULE_INGEST_URL") or
        get_ingest_base_url()
    )

    # Return base URL only - Claude Code appends /v1/traces automatically
    if ingest_url.endswith("/ingest"):
        return ingest_url.replace("/ingest", "")

    return ingest_url.rstrip("/")


def _get_api_key(config: dict[str, str]) -> Optional[str]:
    """Get API key from environment or config."""
    return os.environ.get("ARZULE_API_KEY") or config.get("ARZULE_API_KEY")


def _find_claude_binary() -> Optional[str]:
    """
    Find the Claude CLI binary.
    
    Searches in order:
    1. CLAUDE_CLI_PATH environment variable
    2. 'claude' in PATH
    3. Common installation locations
    """
    # Explicit path
    if os.environ.get("CLAUDE_CLI_PATH"):
        path = os.environ["CLAUDE_CLI_PATH"]
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    
    # Search PATH
    claude = shutil.which("claude")
    if claude:
        return claude
    
    # Common locations
    common_paths = [
        Path.home() / ".local" / "bin" / "claude",
        Path("/usr/local/bin/claude"),
        Path.home() / ".claude" / "bin" / "claude",
    ]
    
    for path in common_paths:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    
    return None


def _build_otel_env(config: dict[str, str]) -> dict[str, str]:
    """
    Build OTel environment variables for Claude Code.
    
    Claude Code supports these OTel environment variables:
    - CLAUDE_CODE_ENABLE_TELEMETRY=1 (enable OTel export)
    - OTEL_EXPORTER_OTLP_ENDPOINT (where to send spans)
    - OTEL_EXPORTER_OTLP_HEADERS (auth headers)
    - OTEL_EXPORTER_OTLP_PROTOCOL (http/json or grpc)
    """
    env = {}
    
    # Enable Claude Code telemetry
    env["CLAUDE_CODE_ENABLE_TELEMETRY"] = "1"
    
    # Set OTLP endpoint
    endpoint = _get_otel_endpoint(config)
    env["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
    
    # Set auth header
    api_key = _get_api_key(config)
    if api_key:
        env["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Bearer {api_key}"
    
    # Use HTTP/JSON protocol (our backend supports this)
    env["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/json"
    
    # Export traces (not just metrics/logs)
    env["OTEL_TRACES_EXPORTER"] = "otlp"
    
    # Service name for resource attributes
    env["OTEL_SERVICE_NAME"] = "claude-code"
    
    return env


def run_claude_with_otel():
    """
    Main entry point - run Claude Code with OTel configured.
    
    This function:
    1. Loads Arzule configuration
    2. Sets up OTel environment variables
    3. Finds the Claude binary
    4. Executes Claude with all arguments passed through
    """
    # Load config
    config = _load_arzule_config()
    
    # Find Claude binary
    claude_path = _find_claude_binary()
    if not claude_path:
        print("Error: Could not find 'claude' CLI.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Make sure Claude Code is installed:", file=sys.stderr)
        print("  https://docs.anthropic.com/en/docs/claude-code", file=sys.stderr)
        print("", file=sys.stderr)
        print("Or set CLAUDE_CLI_PATH to the binary location.", file=sys.stderr)
        sys.exit(1)
    
    # Check for API key
    api_key = _get_api_key(config)
    if not api_key:
        print("Warning: ARZULE_API_KEY not set. OTel data will not be sent.", file=sys.stderr)
        print("Run 'arzule configure' to set up your API key.", file=sys.stderr)
        print("", file=sys.stderr)
    
    # Build environment with OTel vars
    env = os.environ.copy()
    otel_env = _build_otel_env(config)
    env.update(otel_env)
    
    # Pass through any Arzule config vars (for hooks)
    for key, value in config.items():
        if key.startswith("ARZULE_") and key not in env:
            env[key] = value
    
    # Build command: claude + all args
    cmd = [claude_path] + sys.argv[1:]
    
    # Print info if verbose
    if os.environ.get("ARZULE_VERBOSE") or config.get("ARZULE_VERBOSE"):
        print(f"[arzule] OTel endpoint: {otel_env.get('OTEL_EXPORTER_OTLP_ENDPOINT')}", file=sys.stderr)
        print(f"[arzule] Launching: {' '.join(cmd)}", file=sys.stderr)
    
    # Execute Claude (replaces this process)
    try:
        os.execve(claude_path, cmd, env)
    except OSError as e:
        # Fallback to subprocess if execve fails
        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)


def main():
    """CLI entry point."""
    run_claude_with_otel()


if __name__ == "__main__":
    main()


