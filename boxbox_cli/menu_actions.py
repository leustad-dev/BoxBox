"""Placeholder menu action handlers for BoxBox TUI.

Each handler receives a context dict (may include season, round, session)
and returns a short status string to display in the TUI. Some handlers are
still stubs; calendar renders a formatted table.
"""
from __future__ import annotations

from typing import Dict, Optional
from datetime import date, datetime, timezone

import fastf1
import os
import io
from datetime import datetime as _dt


Context = Dict[str, Optional[object]]


# --- Lightweight file logger -------------------------------------------------
_LOG_FILE_PATH: Optional[str] = None


def _ensure_logs_dir() -> str:
    base = os.getcwd()
    logs_dir = os.path.join(base, "logs")
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except Exception:
        pass
    return logs_dir


def _open_log_file() -> str:
    global _LOG_FILE_PATH
    if _LOG_FILE_PATH:
        return _LOG_FILE_PATH
    logs_dir = _ensure_logs_dir()
    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    _LOG_FILE_PATH = os.path.join(logs_dir, f"{ts}.txt")
    # File is created lazily by _log when needed
    return _LOG_FILE_PATH


def _log(line: str) -> None:
    """Write to logfile only for warnings/errors.

    We treat any line containing 'ERROR' or 'WARN' (case-insensitive)
    as worthy of logging. All other lines are ignored to avoid
    unnecessary log file creation and noise.
    """
    try:
        u = str(line).upper()
        if ("ERROR" not in u) and ("WARN" not in u):
            return
        path = _open_log_file()
        # Create the file and write header if it does not exist yet
        header_needed = not os.path.exists(path)
        with open(path, "a", encoding="utf-8") as f:
            if header_needed:
                f.write(f"BoxBox Log started at {_dt.now().isoformat()}\n")
            f.write(line.rstrip("\n") + "\n")
    except Exception:
        pass


def live_timing(ctx: Context) -> str:
    # TODO: wire up fastf1 live timing in future iterations
    return f"Live Timing selected (season={ctx.get('season')}, round={ctx.get('round')}, session={ctx.get('session')})"


def drivers(ctx: Context) -> str:
    """Show drivers grouped by team for a selected year (2018+ only).

    - Uses FastF1 session results (minimal load) to get full driver names and team names.
    - No fallback for <2018 per request; those seasons are not selectable in the UI anymore.
    - Renders a fixed-width table similar to the Calendar view.
    """
    try:
        year = int(ctx.get("season") or date.today().year)
    except Exception:
        year = date.today().year

    title = f"F1 {year} Drivers by Team"

    def clip(val: object, width: int) -> str:
        s = "" if val is None else str(val)
        return s if len(s) <= width else s[:width]

    # Disallow seasons before 2018 (no fallback desired)
    if year < 2018:
        return "Driver lineup data is only available from 2018 onwards."

    # Result collector: list of (team, driver)
    pairs: list[tuple[str, str]] = []

    # Try FastF1 backend for modern seasons (>=2018)
    try:
        # Helper: pick the latest completed session in this season based on the schedule
        def _pick_latest_session_code(year: int) -> Optional[tuple[int, str]]:
            try:
                schedule = fastf1.get_event_schedule(year)
            except Exception:
                return None

            try:
                import pandas as _pd  # type: ignore
            except Exception:  # pragma: no cover
                _pd = None  # type: ignore

            # Collect all sessions with their UTC timestamps
            candidates: list[tuple[datetime, int, str]] = []  # (dt_utc, round, code)

            # Identify name and date columns
            cols = list(getattr(schedule, 'columns', []))
            name_cols = [c for c in cols if c.startswith('Session') and c[-1:].isdigit() and not c.endswith('Utc') and not c.endswith('DateUtc')]

            def _map_name_to_code(name: str) -> Optional[str]:
                n = name.strip().lower()
                if not n:
                    return None
                # Prefer specific competitive sessions
                if 'sprint shootout' in n or n == 'sprint shootout' or 'sprint qualifying' in n:
                    return 'SQ'
                if n == 'sprint' or (('sprint' in n) and ('qual' not in n) and ('shootout' not in n)):
                    return 'S'
                if n == 'qualifying' or (('qualifying' in n) and ('sprint' not in n)):
                    return 'Q'
                if n == 'race' or 'grand prix' in n or 'grandprix' in n:
                    return 'R'
                if 'practice 3' in n or 'fp3' in n:
                    return 'FP3'
                if 'practice 2' in n or 'fp2' in n:
                    return 'FP2'
                if 'practice 1' in n or 'fp1' in n:
                    return 'FP1'
                return None

            # Build candidates list
            for _, row in schedule.iterrows():
                # Round number can be int or str
                rnd_val = row.get('RoundNumber', None)
                try:
                    rnd = int(rnd_val)
                except Exception:
                    # if not convertible, skip (we need numeric round to query)
                    continue
                for nc in name_cols:
                    name_val = str(row.get(nc, '') or '')
                    code = _map_name_to_code(name_val)
                    if not code:
                        continue
                    idx = nc[len('Session'):]
                    dcol = f'Session{idx}DateUtc'
                    if dcol not in cols:
                        continue
                    dt_val = row.get(dcol, None)
                    if dt_val is None:
                        continue
                    dt_utc: Optional[datetime]
                    try:
                        if _pd is not None and isinstance(dt_val, _pd.Timestamp):
                            if dt_val.tzinfo is None:
                                dt_val = dt_val.tz_localize('UTC')
                            dt_utc = dt_val.tz_convert('UTC').to_pydatetime()
                        elif isinstance(dt_val, datetime):
                            dt_utc = dt_val if dt_val.tzinfo else dt_val.replace(tzinfo=timezone.utc)
                        else:
                            continue
                    except Exception:
                        continue
                    candidates.append((dt_utc, rnd, code))

            if not candidates:
                return None

            # Filter to sessions that have started already
            now_utc = datetime.now(timezone.utc)
            candidates = [c for c in candidates if c[0] <= now_utc]
            if not candidates:
                return None

            # Sort by time descending, but enforce session code priority for same timestamp
            priority = {'R': 6, 'Q': 5, 'S': 4, 'SQ': 3, 'FP3': 2, 'FP2': 1, 'FP1': 0}
            candidates.sort(key=lambda x: (x[0], priority.get(x[2], -1)), reverse=True)
            # Take the top candidate
            _, rnd_best, code_best = candidates[0]
            return rnd_best, code_best

        latest = _pick_latest_session_code(year)
        if latest is not None:
            rnd, code = latest
            try:
                sess = fastf1.get_session(year, rnd, code)
                # Minimal load to get results table
                sess.load(telemetry=False, laps=False, weather=False)
                df = getattr(sess, "results", None)
                if df is not None and not getattr(df, "empty", False):
                    name_col = "FullName" if "FullName" in df.columns else (
                        "DriverName" if "DriverName" in df.columns else (
                            "Driver" if "Driver" in df.columns else "Abbreviation"
                        )
                    )
                    team_col = "TeamName" if "TeamName" in df.columns else (
                        "Team" if "Team" in df.columns else None
                    )
                    if team_col is None:
                        # Try loading light laps to map teams
                        try:
                            sess.load(telemetry=False, laps=True, weather=False)
                            laps = getattr(sess, "laps", None)
                            if laps is not None and not laps.empty:
                                latest_laps = laps.groupby("Driver").last()
                                team_map = latest_laps["Team"].to_dict() if "Team" in latest_laps.columns else {}
                                for _, r in df.iterrows():
                                    name = str(r.get(name_col, "")).strip()
                                    abbr = str(r.get("Abbreviation", "")).strip()
                                    team = team_map.get(abbr, "")
                                    if name and team:
                                        pairs.append((team, name))
                        except Exception:
                            pass
                    else:
                        for _, r in df.iterrows():
                            name = str(r.get(name_col, "")).strip()
                            team = str(r.get(team_col, "")).strip()
                            if name and team:
                                pairs.append((team, name))
            except Exception:
                pass
    except Exception:
        pass

    # No fallback requested; if nothing found, return no-data message

    if not pairs:
        return f"No driver/team data Available for {year} F1 Season..."

    # Group by team name
    from collections import defaultdict

    grouped: dict[str, list[str]] = defaultdict(list)
    for team, driver in pairs:
        if driver not in grouped[team]:
            grouped[team].append(driver)

    # Sort teams alphabetically, drivers alphabetically within team
    for t in list(grouped.keys()):
        grouped[t] = sorted(grouped[t], key=lambda s: s.lower())

    teams_sorted = sorted(grouped.keys(), key=lambda s: s.lower())

    # Note for older seasons
    note = None
    if year < 2018:
        note = "Older data before 2018 is not accurate due to API limitation"

    # Compute dynamic column widths based on data
    all_teams = teams_sorted
    all_drivers = [d for t in all_teams for d in grouped.get(t, [])]
    # Header labels
    h_team, h_driver = "Team", "Driver"
    # Determine widths with sensible caps
    def _compute_width(values: list[str], header: str, *, min_w: int, max_w: int) -> int:
        base = max(len(header), max((len(str(v)) for v in values), default=0))
        base = max(min_w, base)
        base = min(max_w, base)
        return base

    TW = _compute_width(all_teams, h_team, min_w=10, max_w=28)
    DW = _compute_width(all_drivers, h_driver, min_w=12, max_w=30)

    header = f"{h_team:<{TW}}  {h_driver:<{DW}}"
    rule_len = max(len(title), len(header))
    hr = "-" * rule_len

    lines: list[str] = ([note] if note else []) + [title, header, hr]
    for team in teams_sorted:
        drivers_list = grouped[team]
        if not drivers_list:
            continue
        # First row with team name
        lines.append(f"{clip(team, TW):<{TW}}  {clip(drivers_list[0], DW):<{DW}}")
        # Subsequent rows: leave team column blank
        for d in drivers_list[1:]:
            lines.append(f"{'':<{TW}}  {clip(d, DW):<{DW}}")
        lines.append(hr)

    return "\n".join(lines)


def results(ctx: Context) -> str:
    """Show season results: driver and constructor standings side-by-side.

    Implementation switched to FastF1-only aggregation (no Ergast), summing
    points from Race ('R') and Sprint ('S') sessions across the selected
    season. This avoids issues with Ergast responses in ongoing seasons.

    - Drivers table: Pos, Driver, Nat, Team, Pts (sorted by points desc)
    - Teams table:   Pos, Team, Pts (sorted by points desc)
    - Seasons prior to 2018 are not supported here (no live timing/results).
    """
    try:
        year = int(ctx.get("season") or date.today().year)
    except Exception:
        year = date.today().year

    title = f"F1 {year} Results"

    # Gutter between the two side-by-side tables
    GUTTER = 4

    def clip(val: object, width: int) -> str:
        s = "" if val is None else str(val)
        return s if len(s) <= width else s[:width]

    # Only log warnings/errors; info logs are suppressed
    _log(f"[results] Rendering results for season={year}")

    if year < 2018:
        _log("[results] WARN: year < 2018 not supported for FastF1 aggregation")
        return "Driver/Team results are only available from 2018 onwards."

    # Aggregation stores
    from collections import defaultdict

    driver_points: dict[str, float] = defaultdict(float)
    driver_name: dict[str, str] = {}
    driver_nat: dict[str, str] = {}
    driver_team: dict[str, str] = {}

    team_points: dict[str, float] = defaultdict(float)

    # Get schedule and iterate sessions up to now
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as exc:
        _log(f"[results] ERROR loading schedule: {exc}")
        return f"No data Available for {year} F1 Season..."

    cols = list(getattr(schedule, "columns", []))
    now_utc = datetime.now(timezone.utc)

    def _row_has_sprint(row) -> bool:
        # Detect sprint weekend via EventFormat or session names
        fmt = str(row.get("EventFormat", "") or "").lower()
        if "sprint" in fmt:
            return True
        # else scan session names
        for c in cols:
            if c.startswith("Session") and not c.endswith("Utc") and not c.endswith("DateUtc") and c[-1:].isdigit():
                name = str(row.get(c, "") or "").lower()
                if name == "sprint" or ("sprint" in name and "qual" not in name and "shootout" not in name):
                    return True
        return False

    def _session_dt_utc(row, code: str):
        # Try to find a matching datetime for code to filter future sessions
        name_cols = [c for c in cols if c.startswith("Session") and c[-1:].isdigit() and not c.endswith("Utc") and not c.endswith("DateUtc")]
        for nc in name_cols:
            idx = nc[len("Session"):]
            dcol = f"Session{idx}DateUtc"
            name = str(row.get(nc, "") or "").strip().lower()
            if code == "R" and (name == "race" or "grand prix" in name or "grandprix" in name):
                return row.get(dcol)
            if code == "S" and (name == "sprint" or ("sprint" in name and "qual" not in name and "shootout" not in name)):
                return row.get(dcol)
        return None

    # Iterate events
    rounds_processed = 0
    sessions_loaded = 0
    for _, row in schedule.iterrows():
        rnd_val = row.get("RoundNumber", None)
        try:
            rnd = int(rnd_val)
        except Exception:
            _log(f"[results] skip row with invalid RoundNumber={rnd_val}")
            continue

        # Determine which sessions to aggregate
        to_consider = ["R"]
        if _row_has_sprint(row):
            to_consider.append("S")

        for code in to_consider:
            # Skip future sessions by timestamp if available
            ts = _session_dt_utc(row, code)
            try:
                # Normalize pandas.Timestamp and naive datetimes
                import pandas as _pd  # type: ignore
                if isinstance(ts, _pd.Timestamp):
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("UTC")
                    ts = ts.tz_convert("UTC").to_pydatetime()
                if isinstance(ts, datetime) and ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                pass
            if ts is not None and isinstance(ts, datetime) and ts > now_utc:
                # future session skipped (info suppressed)
                continue

            try:
                sess = fastf1.get_session(year, rnd, code)
                sess.load(telemetry=False, laps=False, weather=False, messages=False)
                df = getattr(sess, "results", None)
                if df is None or getattr(df, "empty", False):
                    # empty results (info suppressed)
                    continue
                sessions_loaded += 1
                rounds_processed = max(rounds_processed, rnd)

                # Determine useful columns
                cols_r = set(df.columns)
                name_col = (
                    "FullName" if "FullName" in cols_r else
                    ("DriverName" if "DriverName" in cols_r else ("Driver" if "Driver" in cols_r else "Abbreviation"))
                )
                team_col = "TeamName" if "TeamName" in cols_r else ("Team" if "Team" in cols_r else None)
                pts_col = "Points" if "Points" in cols_r else ("points" if "points" in cols_r else None)

                if pts_col is None:
                    _log(f"[results] no Points column for rnd={rnd} code={code}; columns={list(df.columns)}")
                    continue

                for _, r in df.iterrows():
                    abbr = str(r.get("Abbreviation", "") or "").strip()
                    pts = r.get(pts_col, 0) or 0
                    try:
                        pts = float(pts)
                    except Exception:
                        try:
                            pts = float(str(pts).strip())
                        except Exception:
                            pts = 0.0
                    name = str(r.get(name_col, abbr) or abbr)
                    team = str(r.get(team_col, "") or "") if team_col else ""

                    # Nationality via driver metadata when possible
                    nat = driver_nat.get(abbr)
                    if not nat and hasattr(sess, "get_driver") and abbr:
                        try:
                            dmeta = sess.get_driver(abbr) or {}
                            nat = dmeta.get("CountryCode") or dmeta.get("Nationality") or ""
                        except Exception:
                            nat = ""

                    # Update driver stores
                    driver_points[abbr] += pts
                    if name and abbr not in driver_name:
                        driver_name[abbr] = name
                    if nat:
                        driver_nat[abbr] = nat
                    if team:
                        driver_team[abbr] = team

                    # Update team totals
                    if team:
                        team_points[team] += pts

                # aggregated successfully (info suppressed)
            except Exception as exc:
                _log(f"[results] ERROR loading rnd={rnd} code={code}: {exc}")
                continue

    # Prepare drivers table lines
    left_lines: list[str] = []
    # Compute dynamic widths for driver table
    # Gather values for width computation
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
        base = max(min_w, base)
        base = min(max_w, base)
        return base

    POSW = _w([str(i) for i in range(1, max(1, len(driver_rows_preview)) + 1)], "Pos", min_w=2, max_w=3)
    DNAMEW = _w([r["name"] for r in driver_rows_preview], "Driver", min_w=12, max_w=26)
    NATW = _w([r["nat"] for r in driver_rows_preview], "Nat", min_w=3, max_w=4)
    TNAMEW = _w([r["team"] for r in driver_rows_preview], "Team", min_w=10, max_w=26)
    PTSW = _w([str(r["pts"]) for r in driver_rows_preview], "Pts", min_w=3, max_w=6)

    d_header = f"{'Pos':>{POSW}}  {'Driver':<{DNAMEW}}  {'Nat':<{NATW}}  {'Team':<{TNAMEW}}  {'Pts':>{PTSW}}"
    left_lines.append(d_header)
    left_hr = "-" * len(d_header)
    left_lines.append(left_hr)

    # Sort drivers by points desc, then by name
    drivers_sorted = sorted(
        driver_points.items(),
        key=lambda kv: (kv[1], driver_name.get(kv[0], "")),
        reverse=True,
    )

    d_count = 0
    pos_counter = 1
    for abbr, pts in drivers_sorted:
        name = driver_name.get(abbr, abbr)
        nat = driver_nat.get(abbr, "")
        team = driver_team.get(abbr, "")
        left_lines.append(
            f"{pos_counter:>{POSW}}  {clip(name, DNAMEW):<{DNAMEW}}  {clip(nat, NATW):<{NATW}}  {clip(team, TNAMEW):<{TNAMEW}}  {int(pts) if float(pts).is_integer() else round(pts,1):>{PTSW}}"
        )
        d_count += 1
        pos_counter += 1
    if d_count:
        left_lines.append(left_hr)

    # Prepare teams table lines
    right_lines: list[str] = []
    # Compute dynamic widths for constructor table
    team_rows_preview = [{"team": t, "pts": int(p) if float(p).is_integer() else round(p, 1)} for t, p in team_points.items()]
    T_POSW = _w([str(i) for i in range(1, max(1, len(team_rows_preview)) + 1)], "Pos", min_w=2, max_w=3)
    T_TEAMW = _w([r["team"] for r in team_rows_preview], "Team", min_w=10, max_w=28)
    T_PTSW = _w([str(r["pts"]) for r in team_rows_preview], "Pts", min_w=3, max_w=6)

    t_header = f"{'Pos':>{T_POSW}}  {'Team':<{T_TEAMW}}  {'Pts':>{T_PTSW}}"
    right_lines.append(t_header)
    right_hr = "-" * len(t_header)
    right_lines.append(right_hr)

    teams_sorted = sorted(team_points.items(), key=lambda kv: kv[1], reverse=True)
    t_count = 0
    pos_counter = 1
    for team, pts in teams_sorted:
        right_lines.append(
            f"{pos_counter:>{T_POSW}}  {clip(team, T_TEAMW):<{T_TEAMW}}  {int(pts) if float(pts).is_integer() else round(pts,1):>{T_PTSW}}"
        )
        t_count += 1
        pos_counter += 1
    if t_count:
        right_lines.append(right_hr)

    # No data handling
    if d_count == 0 and t_count == 0:
        _log(
            f"[results] WARN: No data aggregated for {year}; rounds_processed={rounds_processed}, sessions_loaded={sessions_loaded}"
        )
        return f"No data Available for {year} F1 Season..."

    # Compose side-by-side lines
    left_width = len(d_header)
    right_width = len(t_header)
    gutter = " " * GUTTER

    lines: list[str] = [title]

    def pad_line(s: str, width: int) -> str:
        if len(s) >= width:
            return s[:width]
        return s + (" " * (width - len(s)))

    max_rows = max(len(left_lines), len(right_lines))
    from itertools import zip_longest
    for l, r in zip_longest(left_lines, right_lines, fillvalue=""):
        lines.append(f"{pad_line(l, left_width)}{gutter}{pad_line(r, right_width)}")

    return "\n".join(lines)


def sessions(ctx: Context) -> str:
    return "Sessions selected (stub)"


def settings(ctx: Context) -> str:
    return "Settings selected (stub)"


def help_about(ctx: Context) -> str:
    return "BoxBox CLI — navigate with ← →, Enter to select, q/ESC to exit"


def calendar(ctx: Context) -> str:
    """Fetch and format the season calendar using FastF1.

    - Shows a fixed-width table with columns:
      Rnd, Country, Location, Sprint Qual, Sprint, Race Qual, Race
    - Uses sprint columns only from 2021 onwards.
    - For seasons < 2018, uses Ergast backend implicitly (or explicitly here)
      and falls back to available date columns.
    """
    try:
        year = int(ctx.get("season") or date.today().year)
    except Exception:
        year = date.today().year

    try:
        backend = None if year >= 2018 else "ergast"
        schedule = fastf1.get_event_schedule(year, backend=backend)
    except Exception as exc:
        return f"Failed to load schedule for {year}: {exc}"

    # If no data is available for the requested year, show a friendly message
    no_data_msg = f"No data Available for {year} F1 Season..."
    try:
        if schedule is None:
            return no_data_msg
        # EventSchedule behaves like a pandas DataFrame
        if hasattr(schedule, "empty") and schedule.empty:
            return no_data_msg
        # Fallback length check
        try:
            if len(schedule) == 0:  # type: ignore[arg-type]
                return no_data_msg
        except Exception:
            pass
    except Exception:
        # If any unexpected error occurs while checking, proceed; later formatting
        # safeguards will still catch the empty case by counting rendered rows.
        pass

    # Optional pandas (comes with fastf1)
    try:
        import pandas as pd  # type: ignore
    except Exception:  # pragma: no cover
        pd = None  # type: ignore

    def to_local(ts):
        try:
            local_tz = datetime.now().astimezone().tzinfo
            if pd is not None and isinstance(ts, pd.Timestamp):
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                return ts.tz_convert(local_tz)
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return ts.astimezone(local_tz)
        except Exception:
            pass
        return ts

    def _abbr_tz(tz_name: Optional[str], dt_obj) -> str:
        try:
            name = (tz_name or "").strip()
            if name:
                if " " in name:
                    parts = [p for p in name.replace("(", " ").replace(")", " ").split() if p.isalpha()]
                    if parts:
                        return "".join(p[0].upper() for p in parts)
                return name
            if hasattr(dt_obj, "utcoffset") and dt_obj.utcoffset() is not None:
                offset = dt_obj.utcoffset()
                total_minutes = int(offset.total_seconds() // 60)
                sign = "+" if total_minutes >= 0 else "-"
                total_minutes = abs(total_minutes)
                hh = total_minutes // 60
                mm = total_minutes % 60
                return f"UTC{sign}{hh:02d}:{mm:02d}"
        except Exception:
            pass
        return "UTC"

    def format_local(dt_obj) -> str:
        loc = to_local(dt_obj)
        try:
            tz_name = loc.tzname() if hasattr(loc, "tzname") else None
            abbr = _abbr_tz(tz_name, loc)
            if hasattr(loc, "strftime"):
                return loc.strftime("%b %d %Y %H:%M ") + abbr
            return str(loc)
        except Exception:
            return str(loc) if loc is not None else ""

    # Title and helpers
    title = f"F1 {year} Calendar"

    def clip(val: object, width: int) -> str:
        s = "" if val is None else str(val)
        return s if len(s) <= width else s[:width]

    # We will build rows first, compute dynamic column widths, then render
    # Column labels
    H_COUNTRY = "Country"
    H_LOCATION = "Location"
    H_SPRINTQ = "Sprint Qual"
    H_SPRINT = "Sprint"
    H_RACEQ = "Race Qual"
    H_RACE = "Race"

    # Dynamic width clamp helper
    def clamp_width(values: list[str], header: str, *, min_w: int, max_w: int) -> int:
        base = max(len(header), max((len(v) for v in values), default=0))
        if base < min_w:
            base = min_w
        if base > max_w:
            base = max_w
        return base

    lines = []

    cols = set(schedule.columns)
    modern = {"RoundNumber", "Country", "Session5DateUtc"}.issubset(cols)

    if modern:
        # Modern schema (2018+ with FastF1 backend). Detect sessions by scanning names
        try:
            events_count = 0
            try:
                schedule_sorted = schedule.sort_values("RoundNumber")
            except Exception:
                schedule_sorted = schedule

            session_name_cols = [
                c for c in schedule_sorted.columns
                if c.startswith("Session") and c[-1:].isdigit() and not c.endswith("Utc") and not c.endswith("DateUtc")
            ]

            def _find_session_dt(row, matchers):
                chosen_idx = None
                for c in sorted(session_name_cols):
                    name = str(row.get(c, ""))
                    if not name:
                        continue
                    lname = name.lower()
                    if any(m(lname) for m in matchers):
                        idx = c[len("Session"):]
                        date_col = f"Session{idx}DateUtc"
                        if date_col in schedule_sorted.columns:
                            chosen_idx = date_col
                            break
                return chosen_idx

            def find_quali_dt(row):
                return _find_session_dt(
                    row,
                    [
                        lambda n: n == "qualifying",
                        lambda n: ("qualifying" in n) and ("sprint" not in n) and ("shootout" not in n),
                    ],
                )

            def find_sprint_dt(row):
                return _find_session_dt(
                    row,
                    [
                        lambda n: n == "sprint",
                        lambda n: ("sprint" in n) and ("qual" not in n) and ("shootout" not in n),
                    ],
                )

            def find_sprint_quali_dt(row):
                return _find_session_dt(
                    row,
                    [
                        lambda n: n == "sprint qualifying",
                        lambda n: n == "sprint shootout",
                        lambda n: "sprint qualifying" in n,
                        lambda n: "sprint shootout" in n,
                    ],
                )

            # First pass: build raw row values
            rows_data: list[dict] = []
            for _, row in schedule_sorted.iterrows():
                rn_val = row.get("RoundNumber", "")
                try:
                    rn = int(rn_val)
                except Exception:
                    rn = rn_val

                country = str(row.get("Country", ""))
                location = str(row.get("Location", "") or "-")
                dt_utc = row.get("Session5DateUtc", None)
                local_str = format_local(dt_utc)

                sprint_str = "-"
                sprint_q_str = "-"
                if year >= 2021:
                    s_col = find_sprint_dt(row)
                    if s_col:
                        sprint_str = format_local(row.get(s_col, None))
                    sq_col = find_sprint_quali_dt(row)
                    if sq_col:
                        sprint_q_str = format_local(row.get(sq_col, None))

                q_col = find_quali_dt(row)
                if q_col:
                    quali_str = format_local(row.get(q_col, None))
                else:
                    quali_str = "-"

                rows_data.append({
                    "rnd": rn,
                    "country": country,
                    "location": location,
                    "sq": sprint_q_str,
                    "s": sprint_str,
                    "q": quali_str,
                    "r": local_str,
                })
                events_count += 1

            # Compute dynamic widths with sensible caps
            CW = clamp_width([str(r["country"]) for r in rows_data], H_COUNTRY, min_w=10, max_w=25)
            LW = clamp_width([str(r["location"]) for r in rows_data], H_LOCATION, min_w=10, max_w=18)
            SQW = clamp_width([str(r["sq"]) for r in rows_data], H_SPRINTQ, min_w=5, max_w=22)
            SW = clamp_width([str(r["s"]) for r in rows_data], H_SPRINT, min_w=5, max_w=22)
            QW = clamp_width([str(r["q"]) for r in rows_data], H_RACEQ, min_w=5, max_w=22)
            RW = clamp_width([str(r["r"]) for r in rows_data], H_RACE, min_w=5, max_w=22)

            header = (
                f"{'Rnd':>3}  {H_COUNTRY:<{CW}}  {H_LOCATION:<{LW}}  "
                f"{H_SPRINTQ:<{SQW}}  {H_SPRINT:<{SW}}  {H_RACEQ:<{QW}}  {H_RACE:<{RW}}"
            )
            rule_len = max(len(header), len(title))
            hr = "-" * rule_len
            lines = [title, header, hr]

            # Render rows with clipping
            for r in rows_data:
                lines.append(
                    f"{r['rnd']:>3}  {clip(r['country'], CW):<{CW}}  {clip(r['location'], LW):<{LW}}  {clip(r['sq'], SQW):<{SQW}}  {clip(r['s'], SW):<{SW}}  {clip(r['q'], QW):<{QW}}  {clip(r['r'], RW):<{RW}}"
                )
                lines.append(hr)
        except Exception as exc:
            return f"Loaded schedule for {year}, but formatting failed: {exc}"

        # If nothing was rendered, inform the user rather than showing an empty table
        if events_count == 0:
            return no_data_msg
        return "\n".join(lines)

    # Legacy/Ergast fallback: try to mimic the same columns with what's available
    try:
        events_count = 0
        try:
            schedule_sorted = schedule.sort_values("RoundNumber")
        except Exception:
            schedule_sorted = schedule

        def find_col_contains(*need):
            for c in schedule_sorted.columns:
                lc = c.lower()
                if all(n in lc for n in need):
                    return c
            return None

        race_date_col = (
            ("RaceDate" if "RaceDate" in schedule_sorted.columns else None)
            or find_col_contains("race", "date")
            or ("EventDate" if "EventDate" in schedule_sorted.columns else None)
            or ("EventStartDate" if "EventStartDate" in schedule_sorted.columns else None)
        )
        quali_date_col = (
            ("QualifyingDate" if "QualifyingDate" in schedule_sorted.columns else None)
            or find_col_contains("qualifying", "date")
        )

        rows_data: list[dict] = []
        for _, row in schedule_sorted.iterrows():
            rn_val = row.get("RoundNumber", "")
            try:
                rn = int(rn_val)
            except Exception:
                rn = rn_val

            country = str(row.get("Country", ""))
            location = str(row.get("Location", "") or "-")

            race_dt = row.get(race_date_col, None) if race_date_col else None
            race_str = format_local(race_dt) if race_dt is not None else "-"

            sprint_str = "-"  # not applicable pre-2021
            sprint_q_str = "-"

            quali_dt = row.get(quali_date_col, None) if quali_date_col else None
            quali_str = format_local(quali_dt) if quali_dt is not None else "-"

            rows_data.append({
                "rnd": rn,
                "country": country,
                "location": location,
                "sq": sprint_q_str,
                "s": sprint_str,
                "q": quali_str,
                "r": race_str,
            })
            events_count += 1

        # Compute dynamic widths with sensible caps
        CW = clamp_width([str(r["country"]) for r in rows_data], H_COUNTRY, min_w=10, max_w=18)
        LW = clamp_width([str(r["location"]) for r in rows_data], H_LOCATION, min_w=10, max_w=18)
        SQW = clamp_width([str(r["sq"]) for r in rows_data], H_SPRINTQ, min_w=5, max_w=22)
        SW = clamp_width([str(r["s"]) for r in rows_data], H_SPRINT, min_w=5, max_w=22)
        QW = clamp_width([str(r["q"]) for r in rows_data], H_RACEQ, min_w=5, max_w=22)
        RW = clamp_width([str(r["r"]) for r in rows_data], H_RACE, min_w=5, max_w=22)

        header = (
            f"{'Rnd':>3}  {H_COUNTRY:<{CW}}  {H_LOCATION:<{LW}}  "
            f"{H_SPRINTQ:<{SQW}}  {H_SPRINT:<{SW}}  {H_RACEQ:<{QW}}  {H_RACE:<{RW}}"
        )
        rule_len = max(len(header), len(title))
        hr = "-" * rule_len
        lines = [title, header, hr]

        # Render rows with clipping
        for r in rows_data:
            lines.append(
                f"{r['rnd']:>3}  {clip(r['country'], CW):<{CW}}  {clip(r['location'], LW):<{LW}}  {clip(r['sq'], SQW):<{SQW}}  {clip(r['s'], SW):<{SW}}  {clip(r['q'], QW):<{QW}}  {clip(r['r'], RW):<{RW}}"
            )
            lines.append(hr)
    except Exception as exc:
        return f"Loaded schedule for {year}, but formatting failed: {exc}"

    if events_count == 0:
        return no_data_msg

    return "\n".join(lines)


# A simple registry that TUI can use if needed
MENU_ACTIONS = {
    "Live": live_timing,
    "Drivers": drivers,
    "Results": results,
    "Sessions": sessions,
    "Calendar": calendar,
    "Settings": settings,
    "Help": help_about,
}
