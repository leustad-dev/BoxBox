"""Dispatcher for BoxBox TUI menu action handlers.

This module re-exports handlers from logical modules in handlers/ and utils/
to provide a stable, centralized API for tui.py.
"""

from __future__ import annotations

from typing import Dict, Optional

# Utilities
from .utils.cache import _enable_ff1_cache
from .utils.stats import (
    prewarm_driver_enrichment,
    complete_season_summaries,
    shutdown_background_tasks,
)

# Handlers
from .handlers.drivers import drivers, driver_stats, constructor_stats
from .handlers.results import results
from .handlers.calendar import calendar
from .handlers.misc import live_timing, sessions, settings, help_about

# Type alias for external use (Context dictionary passed to handlers)
Context = Dict[str, Optional[object]]

__all__ = [
    "Context",
    "drivers",
    "driver_stats",
    "constructor_stats",
    "results",
    "calendar",
    "live_timing",
    "sessions",
    "settings",
    "help_about",
    "prewarm_driver_enrichment",
    "complete_season_summaries",
    "shutdown_background_tasks",
]

# Initialization: Initialize FastF1 cache and logging suppression automatically on import.
_enable_ff1_cache()
