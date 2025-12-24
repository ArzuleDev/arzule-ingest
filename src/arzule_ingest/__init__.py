"""Arzule Ingestion Wrapper - Observability for CrewAI multi-agent systems."""

from .run import ArzuleRun, current_run
from .config import ArzuleConfig

__version__ = "0.1.0"
__all__ = ["ArzuleRun", "current_run", "ArzuleConfig"]

