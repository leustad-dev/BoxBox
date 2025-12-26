from typing import Dict, Optional

Context = Dict[str, Optional[object]]

def live_timing(ctx: Context) -> str:
    return f"Live Timing selected (season={ctx.get('season')}, round={ctx.get('round')}, session={ctx.get('session')})"

def sessions(ctx: Context) -> str:
    return "Sessions selected (stub)"

def settings(ctx: Context) -> str:
    return "Settings selected (stub)"

def help_about(ctx: Context) -> str:
    return "BoxBox CLI — navigate with ← →, Enter to select, q/ESC to exit"
