"""Shared logging helpers for analyser output."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler


_CONSOLE = Console(stderr=True)
_LOGGER_NAME = 'aiida_analyser'


def _configure_package_logger() -> logging.Logger:
    """Create a package logger with Rich formatting if it is not configured yet."""
    logger = logging.getLogger(_LOGGER_NAME)

    if logger.handlers:
        return logger

    handler = RichHandler(
        console=_CONSOLE,
        show_time=False,
        show_level=True,
        show_path=False,
        markup=True,
        rich_tracebacks=False,
    )
    handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``aiida_analyser`` namespace."""
    package_logger = _configure_package_logger()
    if name is None or name in {'', _LOGGER_NAME}:
        return package_logger

    return logging.getLogger(name)


def get_console() -> Console:
    """Return the shared Rich console used by package loggers."""
    _configure_package_logger()
    return _CONSOLE
