"""Centralized endpoint configuration for Arzule SDK.

All backend URLs are defined here to:
1. Hide AWS infrastructure details from public source code
2. Enable easy configuration via environment variables
3. Support future endpoint migrations with single-file changes
"""

from __future__ import annotations

import os
from typing import Optional

# =============================================================================
# BACKEND ENDPOINTS
# =============================================================================

# Primary ingest API (for CrewAI, LangChain, AutoGen SDKs)
# Proxied through Cloudflare to hide AWS API Gateway ID
_DEFAULT_INGEST_BASE = "https://ingest.arzule.com"

# Claude Code integration endpoint (CloudFront distribution)
# Cannot be aliased - we don't control this CloudFront distribution
_DEFAULT_CLAUDE_BASE = "https://dh43xnx5e03pq.cloudfront.net"


def get_ingest_url() -> str:
    """Get the trace ingestion endpoint URL.

    Priority:
    1. ARZULE_INGEST_URL environment variable
    2. Default production endpoint (ingest.arzule.com)

    Returns:
        Full URL including /ingest path
    """
    return os.environ.get("ARZULE_INGEST_URL", f"{_DEFAULT_INGEST_BASE}/ingest")


def get_ingest_base_url() -> str:
    """Get base URL without /ingest path.

    Used for constructing OTEL endpoints and other API paths.

    Returns:
        Base URL without trailing path
    """
    url = os.environ.get("ARZULE_INGEST_URL", _DEFAULT_INGEST_BASE)
    # Strip /ingest suffix if present
    if url.endswith("/ingest"):
        url = url[:-7]
    return url.rstrip("/")


def get_claude_ingest_url() -> str:
    """Get the Claude Code trace ingestion endpoint.

    Priority:
    1. ARZULE_ENDPOINT environment variable
    2. Default CloudFront endpoint

    Note: CloudFront URL cannot be aliased to custom domain
    because we don't control that CloudFront distribution.

    Returns:
        Full URL including /ingest path
    """
    return os.environ.get("ARZULE_ENDPOINT", f"{_DEFAULT_CLAUDE_BASE}/ingest")


def get_attribution_url() -> str:
    """Get the backend attribution matching API endpoint.

    Priority:
    1. ARZULE_ATTRIBUTION_URL environment variable
    2. Derived from ingest base URL + /v1/attribution/match

    Returns:
        Full URL for attribution API
    """
    if url := os.environ.get("ARZULE_ATTRIBUTION_URL"):
        return url

    return f"{get_ingest_base_url()}/v1/attribution/match"


def is_local_endpoint(url: str) -> bool:
    """Check if URL points to localhost (for TLS exemption).

    Args:
        url: URL to check

    Returns:
        True if URL is localhost/127.0.0.1
    """
    return "localhost" in url or "127.0.0.1" in url
