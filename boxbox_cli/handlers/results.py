from __future__ import annotations

import math
from datetime import date
from typing import Dict, Optional
from ..utils.logger import _suppress_ff1_logs, _log, _log_exception
from ..utils.stats import _aggregate_season_stats

Context = Dict[str, Optional[object]]

def results(ctx: Context) -> str:
    """Show season results: driver and constructor standings side-by-side."""
    _suppress_ff1_logs()
    try:
        year = int(ctx.get("season") or date.today().year)
    except Exception:
        year = date.today().year

    title = f"F1 {year} Results"
    GUTTER = 4

    def clip(val: object, width: int) -> str:
        """Clip a string to a specified width."""
        s = "" if val is None else str(val)
        return s if len(s) <= width else s[:width]

    _log(f"[results] Rendering results for season={year}")

    if year < 2018:
        _log("[results] WARN: year < 2018 not supported for FastF1 aggregation")
        return "Driver/Team results are only available from 2018 onwards."

    try:
        drivers_data, teams_data = _aggregate_season_stats(year, basic_only=True)
    except Exception as exc:
        _log_exception(f"results() - aggregation failed for {year}", exc)
        return f"Error loading results for {year}. Please check logs."

    if not drivers_data and not teams_data:
        _log(f"[results] WARN: No data aggregated for {year}")
        return f"No data Available for {year} F1 Season..."

    driver_points = {abbr: d["points"] for abbr, d in drivers_data.items()}
    driver_name = {abbr: d["name"] for abbr, d in drivers_data.items()}
    driver_nat = {abbr: d["nat"] for abbr, d in drivers_data.items()}
    driver_team = {abbr: d["team"] for abbr, d in drivers_data.items()}

    team_points = {team: t["points"] for team, t in teams_data.items()}

    driver_rows_preview = []
    for abbr, pts in driver_points.items():
        driver_rows_preview.append({
            "name": driver_name.get(abbr, abbr),
            "nat": driver_nat.get(abbr, ""),
            "team": driver_team.get(abbr, ""),
            "pts": int(pts) if float(pts).is_integer() else round(pts, 1),
        })

    def _w(values: list[str], header: str, *, min_w: int, max_w: int) -> int:
        base = max(len(header), max((len(str(v)) for v in values), default=0))
        return max(min_w, min(max_w, base))

    POSW = _w([str(i) for i in range(1, len(driver_rows_preview) + 1)], "Pos", min_w=2, max_w=3)
    DNAMEW = _w([r["name"] for r in driver_rows_preview], "Driver", min_w=12, max_w=26)
    NATW = _w([r["nat"] for r in driver_rows_preview], "Nat", min_w=3, max_w=4)
    TNAMEW = _w([r["team"] for r in driver_rows_preview], "Team", min_w=10, max_w=26)
    PTSW = _w([str(r["pts"]) for r in driver_rows_preview], "Pts", min_w=3, max_w=6)

    d_header = f"{'Pos':>{POSW}}  {'Driver':<{DNAMEW}}  {'Nat':<{NATW}}  {'Team':<{TNAMEW}}  {'Pts':>{PTSW}}"
    left_lines = [d_header, "-" * len(d_header)]

    drivers_sorted = sorted(driver_points.items(), key=lambda kv: (kv[1], driver_name.get(kv[0], "")), reverse=True)
    for i, (abbr, pts) in enumerate(drivers_sorted, 1):
        name = driver_name.get(abbr, abbr)
        nat = driver_nat.get(abbr, "")
        team = driver_team.get(abbr, "")
        try:
            fpts = float(pts)
            if math.isnan(fpts): pts_display = "0"
            elif fpts.is_integer(): pts_display = str(int(fpts))
            else: pts_display = str(round(fpts, 1))
        except Exception: pts_display = "0"
        left_lines.append(f"{i:>{POSW}}  {clip(name, DNAMEW):<{DNAMEW}}  {clip(nat, NATW):<{NATW}}  {clip(team, TNAMEW):<{TNAMEW}}  {pts_display:>{PTSW}}")
    if drivers_sorted: left_lines.append("-" * len(d_header))

    team_rows_preview = []
    for t, p in team_points.items():
        try:
            fp = float(p)
            if math.isnan(fp): p_disp = "0"
            elif fp.is_integer(): p_disp = str(int(fp))
            else: p_disp = str(round(fp, 1))
        except Exception: p_disp = "0"
        team_rows_preview.append({"team": t, "pts": p_disp})

    T_POSW = _w([str(i) for i in range(1, len(team_rows_preview) + 1)], "Pos", min_w=2, max_w=3)
    T_TEAMW = _w([r["team"] for r in team_rows_preview], "Team", min_w=10, max_w=28)
    T_PTSW = _w([str(r["pts"]) for r in team_rows_preview], "Pts", min_w=3, max_w=6)

    t_header = f"{'Pos':>{T_POSW}}  {'Team':<{T_TEAMW}}  {'Pts':>{T_PTSW}}"
    right_lines = [t_header, "-" * len(t_header)]

    teams_sorted = sorted(team_points.items(), key=lambda kv: kv[1], reverse=True)
    for i, (team, pts) in enumerate(teams_sorted, 1):
        try:
            fpts = float(pts)
            if math.isnan(fpts): pts_display = "0"
            elif fpts.is_integer(): pts_display = str(int(fpts))
            else: pts_display = str(round(fpts, 1))
        except Exception: pts_display = "0"
        right_lines.append(f"{i:>{T_POSW}}  {clip(team, T_TEAMW):<{T_TEAMW}}  {pts_display:>{T_PTSW}}")
    if teams_sorted: right_lines.append("-" * len(t_header))

    left_width = len(d_header)
    right_width = len(t_header)
    gutter = " " * GUTTER
    lines = [title]

    def pad_line(s: str, width: int) -> str:
        return s[:width] if len(s) >= width else s + (" " * (width - len(s)))

    from itertools import zip_longest
    for l, r in zip_longest(left_lines, right_lines, fillvalue=""):
        lines.append(f"{pad_line(l, left_width)}{gutter}{pad_line(r, right_width)}")

    return "\n".join(lines)
