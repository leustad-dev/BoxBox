from __future__ import annotations

import fastf1
import math
from datetime import date, datetime, timezone
from typing import Dict, Optional, Any
from ..utils.logger import _suppress_ff1_logs, _log_exception
from ..utils.stats import _aggregate_season_stats, _SEASON_STATS_CACHE

try:
    import pandas as pd
except Exception:
    pd = None

Context = Dict[str, Optional[object]]

def drivers(ctx: Context):
    """Show drivers grouped by team for a selected year (2018+ only)."""
    _suppress_ff1_logs()
    try:
        year = int(ctx.get("season") or date.today().year)
    except Exception:
        year = date.today().year

    if year < 2018:
        return "Driver lineup data is only available from 2018 onwards."

    def clip(val: object, width: int) -> str:
        """Clip a string to a specified width."""
        s = "" if val is None else str(val)
        return s if len(s) <= width else s[:width]

    pairs: list[tuple[str, str, str]] = []

    try:
        def _pick_latest_session_code(year: int) -> Optional[tuple[int, str]]:
            """Identify the most recent session that has already occurred for a given year."""
            try:
                schedule = fastf1.get_event_schedule(year)
            except Exception as exc:
                _log_exception(f"drivers() - failed to pick latest session code for {year}", exc)
                return None
            candidates: list[tuple[datetime, int, str]] = []
            cols = list(getattr(schedule, "columns", []))
            name_cols = [c for c in cols if c.startswith("Session") and c[-1:].isdigit() and not c.endswith("Utc") and not c.endswith("DateUtc")]

            def _map_name_to_code(name: str) -> Optional[str]:
                n = name.strip().lower()
                if not n: return None
                if "sprint shootout" in n or n == "sprint shootout" or "sprint qualifying" in n: return "SQ"
                if n == "sprint" or (("sprint" in n) and ("qual" not in n) and ("shootout" not in n)): return "S"
                if n == "qualifying" or (("qualifying" in n) and ("sprint" not in n)): return "Q"
                if n == "race" or "grand prix" in n or "grandprix" in n: return "R"
                if "practice 3" in n or "fp3" in n: return "FP3"
                if "practice 2" in n or "fp2" in n: return "FP2"
                if "practice 1" in n or "fp1" in n: return "FP1"
                return None

            for _, row in schedule.iterrows():
                rnd_val = row.get("RoundNumber", None)
                try: rnd = int(rnd_val)
                except Exception: continue
                for nc in name_cols:
                    name_val = str(row.get(nc, "") or "")
                    code = _map_name_to_code(name_val)
                    if not code: continue
                    idx = nc[len("Session") :]
                    dcol = f"Session{idx}DateUtc"
                    if dcol not in cols: continue
                    dt_val = row.get(dcol, None)
                    if dt_val is None: continue
                    try:
                        if pd is not None and isinstance(dt_val, pd.Timestamp):
                            if dt_val.tzinfo is None: dt_val = dt_val.tz_localize("UTC")
                            dt_utc = dt_val.tz_convert("UTC").to_pydatetime()
                        elif isinstance(dt_val, datetime):
                            dt_utc = dt_val if dt_val.tzinfo else dt_val.replace(tzinfo=timezone.utc)
                        else: continue
                    except Exception: continue
                    candidates.append((dt_utc, rnd, code))

            if not candidates: return None
            now_utc = datetime.now(timezone.utc)
            candidates = [c for c in candidates if c[0] <= now_utc]
            if not candidates: return None
            priority = {"R": 6, "Q": 5, "S": 4, "SQ": 3, "FP3": 2, "FP2": 1, "FP1": 0}
            candidates.sort(key=lambda x: (x[0], priority.get(x[2], -1)), reverse=True)
            return candidates[0][1], candidates[0][2]

        latest = _pick_latest_session_code(year)
        if latest is not None:
            rnd, code = latest
            try:
                sess = fastf1.get_session(year, rnd, code)
                sess.load(telemetry=False, laps=False, weather=False)
                df = getattr(sess, "results", None)
                if df is not None and not getattr(df, "empty", False):
                    name_col = "FullName" if "FullName" in df.columns else ("DriverName" if "DriverName" in df.columns else ("Driver" if "Driver" in df.columns else "Abbreviation"))
                    team_col = "TeamName" if "TeamName" in df.columns else ("Team" if "Team" in df.columns else None)
                    if team_col is None:
                        try:
                            sess.load(telemetry=False, laps=True, weather=False)
                            laps = getattr(sess, "laps", None)
                            if laps is not None and not laps.empty:
                                for _, r in df.iterrows():
                                    abbr = r.get("Abbreviation")
                                    t_laps = laps.pick_drivers(abbr)
                                    team = t_laps["Team"].iloc[0] if not t_laps.empty else "Unknown"
                                    pairs.append((str(team), str(r.get(name_col, abbr)), str(abbr)))
                        except Exception as exc:
                            _log_exception("drivers() - failed to load laps for team identification", exc)
                    else:
                        for _, r in df.iterrows():
                            pairs.append((str(r.get(team_col, "Unknown")), str(r.get(name_col, "Unknown")), str(r.get("Abbreviation", "??"))))
            except Exception as exc:
                _log_exception(f"drivers() - failed to load session {year} rnd={rnd} code={code}", exc)
    except Exception as exc:
        _log_exception(f"drivers() - unexpected error for {year}", exc)

    if not pairs:
        return f"No driver data available for {year}."

    teams_map: Dict[str, list[tuple[str, str]]] = {}
    for t, d_full, d_abbr in pairs:
        teams_map.setdefault(t, []).append((d_full, d_abbr))

    lines: list[str] = [f"F1 {year} Drivers by Team", ""]
    selectables: list[dict] = []

    def _compute_width(values: list[str], header: str, *, min_w: int, max_w: int) -> int:
        base = max(len(header), max((len(str(v)) for v in values), default=0))
        return max(min_w, min(max_w, base))

    all_teams = sorted(teams_map.keys())
    all_driver_names = [d[0] for t_list in teams_map.values() for d in t_list]

    TW = _compute_width(all_teams, "Team", min_w=12, max_w=25)
    DW = _compute_width(all_driver_names, "Driver", min_w=15, max_w=30)

    header = f"{'Team':<{TW}}  {'Driver':<{DW}}"
    lines.append(header)
    hr = "-" * (TW + DW + 2)
    lines.append(hr)

    current_row_index = 4
    for team in all_teams:
        drivers_list = sorted(teams_map[team])
        first_d_name, first_d_abbr = drivers_list[0]
        lines.append(f"{clip(team, TW):<{TW}}  {clip(first_d_name, DW):<{DW}}")
        selectables.append({"row": current_row_index, "type": "team", "team": team})
        selectables.append({"row": current_row_index, "type": "driver", "team": team, "abbr": first_d_abbr, "name": first_d_name})
        current_row_index += 1

        for d_name, d_abbr in drivers_list[1:]:
            lines.append(f"{'':<{TW}}  {clip(d_name, DW):<{DW}}")
            selectables.append({"row": current_row_index, "type": "driver", "team": team, "abbr": d_abbr, "name": d_name})
            current_row_index += 1
        lines.append(hr)
        current_row_index += 1

    colspec = {"team_x": 0, "team_w": TW, "driver_x": TW + 2, "driver_w": DW}
    return {"lines": lines, "selectables": selectables, "season": year, "colspec": colspec}

def driver_stats(ctx: Context, year: int, abbr: str) -> str:
    """Build a stats summary for a driver in a given season (2018+)."""
    _suppress_ff1_logs()
    if year < 2018:
        return "Driver statistics are only available from 2018 onwards."
    try:
        drivers_data, _ = _aggregate_season_stats(year, basic_only=True)
    except Exception as exc:
        _log_exception(f"driver_stats() - aggregation failed for {year}", exc)
        return f"Error loading stats for {year}."
    
    s = drivers_data.get(abbr)
    if not s:
        return f"No data for driver {abbr} in {year}."

    avg_grid = round(s["avg_grid_sum"] / s["avg_grid_cnt"], 1) if s["avg_grid_cnt"] else ""
    avg_finish = round(s["avg_finish_sum"] / s["avg_finish_cnt"], 1) if s["avg_finish_cnt"] else ""

    cache = _SEASON_STATS_CACHE.get(year, {})
    summaries: dict = cache.get("session_summaries", {})
    pit_extra = 0
    fl_extra = 0
    for data in summaries.values():
        try: pit_extra += int(data.get("pitstops", {}).get(abbr, 0))
        except Exception: pass
        try:
            if str(data.get("fastest", "")) == abbr: fl_extra += 1
        except Exception: pass

    total_pits = max(int(s.get("pitstops") or 0), pit_extra)
    total_fl = max(int(s.get("fastest_laps") or 0), fl_extra)

    try:
        pts_val = float(s['points'])
        if math.isnan(pts_val): pts_display = "0"
        elif pts_val.is_integer(): pts_display = str(int(pts_val))
        else: pts_display = str(round(pts_val, 1))
    except Exception: pts_display = "0"

    lines = [
        f"Driver Statistics — {year}",
        "-" * 40,
        f"Name: {s['name']} ({abbr})",
        f"Team: {s['team']}",
        f"Driver No: {s['number']}",
        "",
        f"Total Points: {pts_display}",
        f"Poles: {s['poles']}",
        f"Wins: {s['wins']}",
        f"Podiums: {s['podiums']}",
        f"Fastest Laps: {total_fl}",
        f"Pit Stops: {total_pits}",
        f"Avg Grid: {avg_grid}",
        f"Avg Finish: {avg_finish}",
    ]
    return "\n".join(lines)

def constructor_stats(ctx: Context, year: int, team_name: str) -> str:
    """Build a stats summary for a constructor in a given season (2018+)."""
    _suppress_ff1_logs()
    if year < 2018:
        return "Constructor statistics are only available from 2018 onwards."
    try:
        _, teams_data = _aggregate_season_stats(year)
    except Exception as exc:
        _log_exception(f"constructor_stats() - aggregation failed for {year}", exc)
        return f"Error loading stats for {year}."
    
    s = teams_data.get(team_name)
    if not s:
        return f"No data for constructor {team_name} in {year}."

    lines = [
        f"Constructor Statistics — {year}",
        "-" * 40,
        f"Team: {team_name}",
        "",
        f"Poles: {s['poles']}",
        f"Fastest Laps: {s['fastest_laps']}",
        f"Pit Stops: {s['pitstops']}",
    ]
    return "\n".join(lines)
