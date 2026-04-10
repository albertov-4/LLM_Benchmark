"""Centralized logging helpers for the LLM-Needs-a-Plan project."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

_DEFAULT_FORMAT = "%(asctime)s | %(levelname).1s | %(name)s | %(message)s"
_DEFAULT_DATE = "%Y-%m-%d %H:%M:%S"


def _coerce_level(level: Union[str, int, None]) -> int:
    """Return a numeric logging level from flexible input."""
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        if isinstance(resolved, int):
            return resolved
    return logging.INFO


def configure_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATE,
) -> None:
    """Configure root logging with optional file output."""
    numeric_level = _coerce_level(level)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        try:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(path, mode="a", encoding="utf-8"))
        except OSError:
            # Fall back to stdout-only if we cannot create the file handler.
            pass

    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )

    logging.captureWarnings(True)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a namespaced logger with project defaults."""
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
