"""
BoxBox CLI entry point using Typer.

Run:
  python cli.py live

This starts a curses-based TUI with a top menu bar navigable via arrow keys.
"""
from __future__ import annotations

import typer

from boxbox_cli.tui import run_tui


app = typer.Typer(add_completion=False, help="BoxBox - Formula 1 live CLI (framework)")


@app.command()
def live(
    season: int = typer.Option(None, help="Season year, e.g. 2025 (optional for now)"),
    round: int = typer.Option(None, help="Round number, e.g. 23 (optional for now)"),
    session: str = typer.Option(
        None, help="Session code (FP1, FP2, FP3, Q, SQ, R, etc.) (optional for now)"
    ),
):
    """Start the interactive TUI."""
    # Pass context to the TUI; for now we keep it simple and only pass args.
    run_tui({"season": season, "round": round, "session": session})


if __name__ == "__main__":
    app()
