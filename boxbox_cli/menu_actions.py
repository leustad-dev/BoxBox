"""Placeholder menu action handlers for BoxBox TUI.

Each handler receives a context dict (may include season, round, session)
and returns a short status string to display in the TUI. Some handlers are
still stubs; calendar renders a formatted table.
"""

from __future__ import annotations

from typing import Dict, Optional
from datetime import date, datetime, timezone

import math
import fastf1
import os
import io
from datetime import datetime as _dt

# Non-problematic third-party dependency imported at module level
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


# Enable FastF1 on-disk cache to speed up repeated session loads (best-effort).
# Honor FASTF1_CACHE_DIR env var; default to project_root\cache
try:  # pragma: no cover - environment dependent
    from fastf1 import Cache as _FF1Cache  # type: ignore

    def _enable_ff1_cache() -> None:
        # Suppress FastF1 logs to avoid corrupting the TUI.
        _suppress_ff1_logs()

        cache_dir = os.getenv("FASTF1_CACHE_DIR")
        if not cache_dir:
            # Default to project root / cache
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_dir = os.path.join(project_root, "cache")
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception:
            # If creating fails, let FastF1 try anyway
            pass
        try:
            # Guard against double-initialization when possible
            is_enabled = getattr(_FF1Cache, "is_enabled", None)
            get_path = getattr(_FF1Cache, "get_cache_path", None)
            if callable(is_enabled) and callable(get_path):
                try:
                    if is_enabled() and str(get_path()) == str(cache_dir):
                        return
                except Exception:
                    pass
            _FF1Cache.enable_cache(cache_dir)
        except Exception:
            # Silently ignore; FastF1 will fall back to its defaults
            pass

    _enable_ff1_cache()
except Exception:
    pass


Context = Dict[str, Optional[object]]


# --- Lightweight file logger -------------------------------------------------
_LOG_FILE_PATH: Optional[str] = None


def _suppress_ff1_logs() -> None:
    """Ensure FastF1 loggers do not output to the console and redirect to file.
    
    FastF1 uses its own internal LoggingManager which sets up a StreamHandler
    on the 'fastf1' logger. We remove it and add a FileHandler.
    """
    try:
        import logging
        import fastf1
        
        # Ensure the logs directory exists and get a file path
        logs_dir = _ensure_logs_dir()
        # We reuse the same log file for the session if already created
        log_file = _open_log_file()
        
        ff1_logger = logging.getLogger("fastf1")
        
        # 1. Silence the logger itself and its children at the logger level
        # We set it to WARNING to capture important stuff in the file, 
        # but we will control the output via handlers.
        ff1_logger.setLevel(logging.WARNING)
        fastf1.set_log_level("WARNING")
        
        # 2. Redirect output: Remove StreamHandlers and add FileHandler
        # FastF1's set_log_level often re-adds or modifies a StreamHandler.
        
        # Check if we already have our FileHandler to avoid duplicates
        has_file_handler = any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file) for h in ff1_logger.handlers)
        
        if not has_file_handler:
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setLevel(logging.WARNING)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ff1_logger.addHandler(fh)
            
        # 3. Remove or silence any StreamHandlers (console output)
        for h in ff1_logger.handlers[:]:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                # We can either remove it or set its level to CRITICAL+1 to silence it
                # Removing is cleaner but FastF1 might re-add it.
                # Setting level to a very high value is safer.
                h.setLevel(100) 
                # ff1_logger.removeHandler(h) # Uncomment if we prefer removal
                
        # 4. Force set level for all currently existing fastf1 loggers
        for name in logging.root.manager.loggerDict:
            if name.startswith("fastf1"):
                logging.getLogger(name).setLevel(logging.WARNING)
                
    except Exception:
        pass


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


def drivers(ctx: Context):
    """Show drivers grouped by team for a selected year (2018+ only).

    Returns either a string (error/no-data) or a dict with:
      { 'lines': list[str], 'selectables': list[dict], 'season': int }

    - Uses FastF1 session results (minimal load) to get full driver names and team names.
    - No fallback for <2018 per request; those seasons are not selectable in the UI anymore.
    - Renders a fixed-width table similar to the Calendar view and includes row metadata
      so the TUI can enable selection of teams and drivers.
    """
    _suppress_ff1_logs()
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

    # Result collector: list of (team, driver_full_name, driver_abbr)
    pairs: list[tuple[str, str, str]] = []

    # Try FastF1 backend for modern seasons (>=2018)
    try:
        # Helper: pick the latest completed session in this season based on the schedule
        def _pick_latest_session_code(year: int) -> Optional[tuple[int, str]]:
            try:
                schedule = fastf1.get_event_schedule(year)
            except Exception:
                return None

            # Collect all sessions with their UTC timestamps
            candidates: list[tuple[datetime, int, str]] = []  # (dt_utc, round, code)

            # Identify name and date columns
            cols = list(getattr(schedule, "columns", []))
            name_cols = [
                c
                for c in cols
                if c.startswith("Session")
                and c[-1:].isdigit()
                and not c.endswith("Utc")
                and not c.endswith("DateUtc")
            ]

            def _map_name_to_code(name: str) -> Optional[str]:
                n = name.strip().lower()
                if not n:
                    return None
                # Prefer specific competitive sessions
                if (
                    "sprint shootout" in n
                    or n == "sprint shootout"
                    or "sprint qualifying" in n
                ):
                    return "SQ"
                if n == "sprint" or (
                    ("sprint" in n) and ("qual" not in n) and ("shootout" not in n)
                ):
                    return "S"
                if n == "qualifying" or (("qualifying" in n) and ("sprint" not in n)):
                    return "Q"
                if n == "race" or "grand prix" in n or "grandprix" in n:
                    return "R"
                if "practice 3" in n or "fp3" in n:
                    return "FP3"
                if "practice 2" in n or "fp2" in n:
                    return "FP2"
                if "practice 1" in n or "fp1" in n:
                    return "FP1"
                return None

            # Build candidates list
            for _, row in schedule.iterrows():
                # Round number can be int or str
                rnd_val = row.get("RoundNumber", None)
                try:
                    rnd = int(rnd_val)
                except Exception:
                    # if not convertible, skip (we need numeric round to query)
                    continue
                for nc in name_cols:
                    name_val = str(row.get(nc, "") or "")
                    code = _map_name_to_code(name_val)
                    if not code:
                        continue
                    idx = nc[len("Session") :]
                    dcol = f"Session{idx}DateUtc"
                    if dcol not in cols:
                        continue
                    dt_val = row.get(dcol, None)
                    if dt_val is None:
                        continue
                    dt_utc: Optional[datetime]
                    try:
                        if pd is not None and isinstance(dt_val, pd.Timestamp):
                            if dt_val.tzinfo is None:
                                dt_val = dt_val.tz_localize("UTC")
                            dt_utc = dt_val.tz_convert("UTC").to_pydatetime()
                        elif isinstance(dt_val, datetime):
                            dt_utc = (
                                dt_val
                                if dt_val.tzinfo
                                else dt_val.replace(tzinfo=timezone.utc)
                            )
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
            priority = {"R": 6, "Q": 5, "S": 4, "SQ": 3, "FP3": 2, "FP2": 1, "FP1": 0}
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
                    name_col = (
                        "FullName"
                        if "FullName" in df.columns
                        else (
                            "DriverName"
                            if "DriverName" in df.columns
                            else (
                                "Driver" if "Driver" in df.columns else "Abbreviation"
                            )
                        )
                    )
                    team_col = (
                        "TeamName"
                        if "TeamName" in df.columns
                        else ("Team" if "Team" in df.columns else None)
                    )
                    if team_col is None:
                        # Try loading light laps to map teams
                        try:
                            sess.load(telemetry=False, laps=True, weather=False)
                            laps = getattr(sess, "laps", None)
                            if laps is not None and not laps.empty:
                                latest_laps = laps.groupby("Driver").last()
                                team_map = (
                                    latest_laps["Team"].to_dict()
                                    if "Team" in latest_laps.columns
                                    else {}
                                )
                                for _, r in df.iterrows():
                                    name = str(r.get(name_col, "")).strip()
                                    abbr = str(r.get("Abbreviation", "")).strip()
                                    team = team_map.get(abbr, "")
                                    if name and team:
                                        pairs.append((team, name, abbr))
                        except Exception:
                            pass
                    else:
                        for _, r in df.iterrows():
                            name = str(r.get(name_col, "")).strip()
                            team = str(r.get(team_col, "")).strip()
                            abbr = str(r.get("Abbreviation", "")).strip()
                            if name and team:
                                pairs.append((team, name, abbr))
            except Exception:
                pass
    except Exception:
        pass

    # No fallback requested; if nothing found, return no-data message

    if not pairs:
        return f"No driver/team data Available for {year} F1 Season..."

    # Group by team name
    from collections import defaultdict

    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for team, driver, abbr in pairs:
        if all(driver != d for d, _ in grouped[team]):
            grouped[team].append((driver, abbr))

    # Sort teams alphabetically, drivers alphabetically within team
    for t in list(grouped.keys()):
        grouped[t] = sorted(grouped[t], key=lambda x: x[0].lower())

    teams_sorted = sorted(grouped.keys(), key=lambda s: s.lower())

    # Note for older seasons
    note = None
    if year < 2018:
        note = "Older data before 2018 is not accurate due to API limitation"

    # Compute dynamic column widths based on data
    all_teams = teams_sorted
    all_drivers = [d for t in all_teams for (d, _) in grouped.get(t, [])]
    # Header labels
    h_team, h_driver = "Team", "Driver"

    # Determine widths with sensible caps
    def _compute_width(
        values: list[str], header: str, *, min_w: int, max_w: int
    ) -> int:
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
    selectables: list[dict] = []
    current_row_index = len(lines)  # first row we will append gets this index

    for team in teams_sorted:
        drivers_list = grouped[team]
        if not drivers_list:
            continue
        # First row with team name
        first_driver, first_abbr = drivers_list[0]
        lines.append(f"{clip(team, TW):<{TW}}  {clip(first_driver, DW):<{DW}}")
        # Mark team row selectable as constructor
        selectables.append({
            "row": current_row_index,
            "type": "team",
            "team": team,
        })
        current_row_index += 1

        # Also mark first driver row selectable
        selectables.append({
            "row": current_row_index - 1,  # same line has driver too
            "type": "driver",
            "team": team,
            "abbr": first_abbr,
            "name": first_driver,
        })

        # Subsequent rows: leave team column blank
        for d_name, d_abbr in drivers_list[1:]:
            lines.append(f"{'':<{TW}}  {clip(d_name, DW):<{DW}}")
            selectables.append({
                "row": current_row_index,
                "type": "driver",
                "team": team,
                "abbr": d_abbr,
                "name": d_name,
            })
            current_row_index += 1

        lines.append(hr)
        current_row_index += 1

    # Provide column spec so the TUI can highlight only the selected cell
    colspec = {
        "team_x": 0,
        "team_w": TW,
        # two spaces between columns in the rendered line
        "driver_x": TW + 2,
        "driver_w": DW,
    }

    return {"lines": lines, "selectables": selectables, "season": year, "colspec": colspec}


_SEASON_STATS_CACHE: dict[int, dict] = {}


def _aggregate_season_stats(year: int, *, basic_only: bool = True):
    """Aggregate season-wide stats for drivers and teams (2018+).

    Returns (drivers, teams) where:
      drivers: dict[abbr] -> {
        name, nat, dob, number, team,
        wins, podiums, poles, fastest_laps, pitstops,
        avg_grid_sum, avg_grid_cnt, avg_finish_sum, avg_finish_cnt,
        points
      }
      teams: dict[team] -> {wins, podiums, poles, fastest_laps, pitstops, avg_finish_sum, avg_finish_cnt}
    """
    _suppress_ff1_logs()
    # Cache hit
    cached = _SEASON_STATS_CACHE.get(year)
    if cached and cached.get("basic_ready"):
        return cached["drivers"], cached["teams"]

    from collections import defaultdict

    drivers: dict[str, dict] = defaultdict(
        lambda: {
            "name": "",
            "nat": "",
            "dob": None,
            "number": "",
            "team": "",
            "wins": 0,
            "podiums": 0,
            "poles": 0,
            "fastest_laps": 0,
            "pitstops": 0,
            "avg_grid_sum": 0.0,
            "avg_grid_cnt": 0,
            "avg_finish_sum": 0.0,
            "avg_finish_cnt": 0,
            "points": 0.0,
        }
    )
    teams: dict[str, dict] = defaultdict(
        lambda: {
            "poles": 0,
            "wins": 0,
            "podiums": 0,
            "fastest_laps": 0,
            "pitstops": 0,
            "avg_finish_sum": 0.0,
            "avg_finish_cnt": 0,
            "points": 0.0,
        }
    )

    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as exc:
        _log(f"[stats] ERROR loading schedule {year}: {exc}")
        return drivers, teams

    cols = list(getattr(schedule, "columns", []))
    now_utc = datetime.now(timezone.utc)

    def _session_dt_utc(row, code: str):
        name_cols = [
            c
            for c in cols
            if c.startswith("Session")
            and c[-1:].isdigit()
            and not c.endswith("Utc")
            and not c.endswith("DateUtc")
        ]
        for nc in name_cols:
            idx = nc[len("Session") :]
            dcol = f"Session{idx}DateUtc"
            name = str(row.get(nc, "") or "").strip().lower()
            if code == "R" and (name == "race" or "grand prix" in name or "grandprix" in name):
                return row.get(dcol)
            if code == "Q" and name == "qualifying":
                return row.get(dcol)
            if code == "S" and (name == "sprint" or ("sprint" in name and "qual" not in name and "shootout" not in name)):
                return row.get(dcol)
        return None

    for _, row in schedule.iterrows():
        rnd_val = row.get("RoundNumber", None)
        try:
            rnd = int(rnd_val)
        except Exception:
            continue

        # Qualifying for poles
        qts = _session_dt_utc(row, "Q")
        try:
            if pd is not None and isinstance(qts, pd.Timestamp):
                if qts.tzinfo is None:
                    qts = qts.tz_localize("UTC")
                qts = qts.tz_convert("UTC").to_pydatetime()
        except Exception:
            pass
        if isinstance(qts, datetime) and qts <= now_utc:
            try:
                q = fastf1.get_session(year, rnd, "Q")
                q.load(telemetry=False, laps=False, weather=False, messages=False)
                dfq = getattr(q, "results", None)
                if dfq is not None and not getattr(dfq, "empty", False):
                    pole = dfq.nsmallest(1, "Position").iloc[0]
                    ab = str(pole.get("Abbreviation", "") or "").strip()
                    tm = str(pole.get("TeamName", pole.get("Team", "")) or "")
                    if ab:
                        drivers[ab]["poles"] += 1
                    if tm:
                        teams[tm]["poles"] += 1
            except Exception as exc:
                _log(f"[stats] WARN: qual rnd={rnd} failed: {exc}")

        # Race + maybe sprint
        to_consider = ["R"]
        fmt = str(row.get("EventFormat", "") or "").lower()
        if "sprint" in fmt:
            to_consider.append("S")
        else:
            # Fallback: scan session names for "sprint"
            for nc in [c for c in cols if c.startswith("Session") and c[-1:].isdigit() and not c.endswith("Utc")]:
                sn = str(row.get(nc, "") or "").lower()
                if "sprint" in sn and "qual" not in sn and "shootout" not in sn:
                    to_consider.append("S")
                    break

        for code in to_consider:
            ts = _session_dt_utc(row, code)
            try:
                if pd is not None and isinstance(ts, pd.Timestamp):
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("UTC")
                    ts = ts.tz_convert("UTC").to_pydatetime()
            except Exception:
                pass
            if not (isinstance(ts, datetime) and ts <= now_utc):
                continue

            try:
                s = fastf1.get_session(year, rnd, code)
                # For basic aggregation keep it light: no laps
                s.load(telemetry=False, laps=not basic_only, weather=False, messages=False)
                dfr = getattr(s, "results", None)
                if dfr is None or getattr(dfr, "empty", False):
                    continue
                fl_abbr = ""
                if not basic_only:
                        # Only compute fastest laps in heavy mode
                        try:
                            fl = s.laps.pick_fastest()
                            fl_abbr = str(fl.get("Driver", "") or "").strip()
                        except Exception:
                            fl_abbr = ""

                for _, r in dfr.iterrows():
                    abbr = str(r.get("Abbreviation", "") or "").strip()
                    name = r.get("FullName") or r.get("DriverName") or abbr
                    team = r.get("TeamName") or r.get("Team") or ""
                    nat = r.get("Nationality") or ""
                    num = r.get("DriverNumber") or ""
                    pos = r.get("Position")
                    grid = r.get("GridPosition")
                    pts = r.get("Points", 0) or 0
                    try:
                        pts = float(pts)
                        if math.isnan(pts):
                            pts = 0.0
                    except Exception:
                        try:
                            pts = float(str(pts).strip())
                            if math.isnan(pts):
                                pts = 0.0
                        except Exception:
                            pts = 0.0

                    d = drivers[abbr]
                    if name:
                        d["name"] = name
                    if team:
                        d["team"] = team
                    if nat:
                        d["nat"] = d["nat"] or nat
                    if num and not d["number"]:
                        d["number"] = str(num)
                    # date of birth
                    # enrich from driver metadata
                    try:
                        meta = s.get_driver(abbr) or {}
                        # nationality fallbacks
                        if not nat:
                            nat = (
                                meta.get("Nationality")
                                or meta.get("CountryCode")
                                or meta.get("Country")
                                or ""
                            )
                        if not d["nat"] and nat:
                            d["nat"] = nat
                        # date of birth
                        if not d["dob"]:
                            d["dob"] = (
                                meta.get("DateOfBirth")
                                or meta.get("DOB")
                                or meta.get("BirthDate")
                            )
                    except Exception:
                        pass

                    try:
                        if math.isnan(float(pos)):
                            ipos = 0
                        else:
                            ipos = int(pos)
                        if ipos == 1:
                            d["wins"] += 1
                        if 1 <= ipos <= 3:
                            d["podiums"] += 1
                        if ipos > 0:
                            d["avg_finish_sum"] += ipos
                            d["avg_finish_cnt"] += 1
                    except Exception:
                        pass

                    try:
                        if math.isnan(float(grid)):
                            igrid = 0
                        else:
                            igrid = int(grid)
                        if igrid > 0:
                            d["avg_grid_sum"] += igrid
                            d["avg_grid_cnt"] += 1
                    except Exception:
                        pass

                    # accumulate season points
                    d["points"] += pts
                    if team:
                        teams[team]["points"] += pts

                    # pit stops only in heavy mode to avoid loading laps by default
                    if not basic_only:
                        try:
                            dlaps = s.laps.pick_drivers(abbr)
                            d["pitstops"] += int(dlaps["PitInTime"].notna().sum())
                        except Exception:
                            pass

                    if team:
                        try:
                            fpos = float(pos)
                            if code == "R" and not math.isnan(fpos):
                                teams[team]["avg_finish_sum"] += int(fpos)
                                teams[team]["avg_finish_cnt"] += 1
                        except Exception:
                            pass
                        if code == "R" and not basic_only:
                            try:
                                teams[team]["pitstops"] += int(dlaps["PitInTime"].notna().sum())
                            except Exception:
                                pass

                if fl_abbr:
                    drivers[fl_abbr]["fastest_laps"] += 1
                    # count for team of fl driver
                    try:
                        tname = drivers[fl_abbr]["team"]
                        if tname:
                            teams[tname]["fastest_laps"] += 1
                    except Exception:
                        pass
            except Exception as exc:
                _log(f"[stats] WARN: load rnd={rnd} code={code} failed: {exc}")

    # Save in cache
    _SEASON_STATS_CACHE[year] = {
        "drivers": drivers,
        "teams": teams,
        "basic_ready": True,
        "enriched": set(),  # type: ignore
        # session_summaries: (round, code) -> { 'pitstops': {abbr: int}, 'fastest': abbr }
        "session_summaries": {},
        "processed_sessions": set(),
    }
    return drivers, teams


def _ensure_session_summary(year: int, rnd: int, code: str) -> Optional[dict]:
    """Load a session once and compute summary:
    returns { 'pitstops': {abbr: int}, 'fastest': abbr_or_'' }
    Caches the result in _SEASON_STATS_CACHE[year]['session_summaries'].
    """
    _suppress_ff1_logs()
    cache = _SEASON_STATS_CACHE.get(year)
    if not cache:
        _aggregate_season_stats(year, basic_only=True)
        cache = _SEASON_STATS_CACHE.get(year)
    if not cache:
        return None
    summaries: dict = cache.setdefault("session_summaries", {})
    key = (int(rnd), str(code))
    if key in summaries:
        return summaries[key]
    try:
        s = fastf1.get_session(year, rnd, code)
        s.load(telemetry=False, laps=True, weather=False, messages=False)
        laps = getattr(s, "laps", None)
        if laps is None or laps.empty:
            res = {"pitstops": {}, "fastest": ""}
            summaries[key] = res
            cache["session_summaries"] = summaries
            return res
        # Pitstops per driver: count laps with PitInTime present
        try:
            pit_series = laps["PitInTime"].notna()
            pit_counts = laps.loc[pit_series, ["Driver", "PitInTime"]].groupby("Driver").size()
            pit_map = {str(k): int(v) for k, v in pit_counts.to_dict().items()}
        except Exception:
            pit_map = {}
        # Fastest lap owner
        fastest = ""
        try:
            fl = laps.pick_fastest()
            fastest = str(fl.get("Driver", "") or "").strip()
        except Exception:
            fastest = ""
        res = {"pitstops": pit_map, "fastest": fastest}
        summaries[key] = res
        cache["session_summaries"] = summaries
        # mark processed
        processed: set = cache.setdefault("processed_sessions", set())  # type: ignore
        processed.add(key)
        cache["processed_sessions"] = processed
        return res
    except Exception as exc:
        _log(f"[summary] WARN: load summary rnd={rnd} code={code} failed: {exc}")
        return None


def _enrich_driver(year: int, abbr: str) -> None:
    """Lazily enrich a single driver's stats using precomputed session summaries.

    Does not iterate all sessions if summaries are not available; only sums what we have.
    """
    cache = _SEASON_STATS_CACHE.get(year)
    if not cache:
        _aggregate_season_stats(year, basic_only=True)
        cache = _SEASON_STATS_CACHE.get(year) or {}
    drivers = cache.get("drivers", {})
    enriched: set = cache.get("enriched", set())  # type: ignore
    if abbr in enriched:
        return
    summaries: dict = cache.get("session_summaries", {})
    # Sum existing summaries
    pitstops = 0
    fastest_laps = 0
    for (_rnd, _code), data in list(summaries.items()):
        try:
            pitstops += int(data.get("pitstops", {}).get(abbr, 0))
        except Exception:
            pass
        try:
            if str(data.get("fastest", "")) == abbr:
                fastest_laps += 1
        except Exception:
            pass
    # Write back what we have without blocking
    d = drivers.get(abbr)
    if d is not None:
        if pitstops:
            d["pitstops"] = pitstops
        if fastest_laps:
            d["fastest_laps"] = fastest_laps
    enriched.add(abbr)
    cache["enriched"] = enriched


def prewarm_driver_enrichment(year: int, abbrs: list[str]) -> None:
    """Precompute summaries for a limited set of recently completed sessions to
    reduce first-popup latency. Processes only Race sessions for the last N rounds
    (default N=5) and updates cached driver pitstops/fastest-lap counts.

    Idempotent across calls; also safe to run multiple times.
    """
    _suppress_ff1_logs()
    if year < 2018:
        return
    # Ensure basic cache
    _aggregate_season_stats(year, basic_only=True)
    cache = _SEASON_STATS_CACHE.get(year)
    if not cache:
        return
    drivers = cache.get("drivers", {})
    enriched: set = cache.get("enriched", set())  # type: ignore
    processed = cache.get("processed_sessions", set())  # type: ignore
    if not isinstance(processed, set):
        processed = set()

    target_abbrs = {a for a in abbrs if isinstance(a, str) and a}
    if not target_abbrs:
        cache["processed_sessions"] = processed
        return

    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as exc:
        _log(f"[prewarm] WARN: schedule {year} failed: {exc}")
        return

    cols = list(getattr(schedule, "columns", []))
    now_utc = datetime.now(timezone.utc)

    def _session_dt_utc(row, code: str):
        name_cols = [
            c
            for c in cols
            if c.startswith("Session")
            and c[-1:].isdigit()
            and not c.endswith("Utc")
            and not c.endswith("DateUtc")
        ]
        for nc in name_cols:
            idx = nc[len("Session") :]
            dcol = f"Session{idx}DateUtc"
            name = str(row.get(nc, "") or "").strip().lower()
            if code == "R" and (name == "race" or "grand prix" in name or "grandprix" in name):
                return row.get(dcol)
            if code == "S" and (name == "sprint" or ("sprint" in name and "qual" not in name and "shootout" not in name)):
                return row.get(dcol)
        return None

    # Collect completed race sessions only
    completed_races: list[tuple[int, str, datetime]] = []
    for _, row in schedule.iterrows():
        rnd_val = row.get("RoundNumber", None)
        try:
            rnd = int(rnd_val)
        except Exception:
            continue
        ts = _session_dt_utc(row, "R")
        try:
            if pd is not None and isinstance(ts, pd.Timestamp):
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                ts = ts.tz_convert("UTC").to_pydatetime()
        except Exception:
            pass
        if isinstance(ts, datetime) and ts <= now_utc:
            completed_races.append((rnd, "R", ts))

    # Limit to last N races
    completed_races.sort(key=lambda x: x[2])
    LAST_N = 5
    subset = completed_races[-LAST_N:]

    # Process with bounded parallelism
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
    except Exception:
        ThreadPoolExecutor = None  # type: ignore
    tasks = []
    if ThreadPoolExecutor is not None:
        with ThreadPoolExecutor(max_workers=3) as ex:
            for rnd, code, _ in subset:
                key = (rnd, code)
                if key in processed:
                    continue
                tasks.append(ex.submit(_ensure_session_summary, year, rnd, code))
            for fut in tasks:
                try:
                    fut.result(timeout=None)
                except Exception as exc:
                    _log(f"[prewarm] WARN: summary task failed: {exc}")
    else:
        for rnd, code, _ in subset:
            key = (rnd, code)
            if key in processed:
                continue
            _ensure_session_summary(year, rnd, code)

    # Apply summaries to target drivers
    summaries: dict = cache.get("session_summaries", {})
    for (rnd, code) in [(r, c) for (r, c, _) in subset]:
        data = summaries.get((rnd, code), {})
        pit_map = data.get("pitstops", {}) or {}
        fl = str(data.get("fastest", "") or "")
        for ab in list(target_abbrs):
            try:
                drivers[ab]["pitstops"] += int(pit_map.get(ab, 0))
            except Exception:
                pass
        if fl:
            try:
                drivers[fl]["fastest_laps"] += 1
            except Exception:
                pass
        processed.add((rnd, code))

    cache["processed_sessions"] = processed
    # Do not force-set enriched; allow detailed enrichment to be completed in background


def driver_stats(ctx: Context, year: int, abbr: str) -> str:
    """Build a stats summary for a driver in a given season (2018+)."""
    _suppress_ff1_logs()
    if year < 2018:
        return "Driver statistics are only available from 2018 onwards."
    drivers, _ = _aggregate_season_stats(year, basic_only=True)
    s = drivers.get(abbr)
    if not s:
        return f"No data for driver {abbr} in {year}."

    avg_grid = (
        round(s["avg_grid_sum"] / s["avg_grid_cnt"], 1) if s["avg_grid_cnt"] else ""
    )
    avg_finish = (
        round(s["avg_finish_sum"] / s["avg_finish_cnt"], 1)
        if s["avg_finish_cnt"]
        else ""
    )

    # Prefer non-blocking enrichment: sum from session_summaries if available
    cache = _SEASON_STATS_CACHE.get(year, {})
    summaries: dict = cache.get("session_summaries", {})
    pit_extra = 0
    fl_extra = 0
    for (_k, data) in summaries.items():
        try:
            pit_extra += int(data.get("pitstops", {}).get(abbr, 0))
        except Exception:
            pass
        try:
            if str(data.get("fastest", "")) == abbr:
                fl_extra += 1
        except Exception:
            pass
    # Merge extras without blocking
    total_pits = int(s.get("pitstops") or 0)
    total_pits = max(total_pits, pit_extra)
    total_fl = int(s.get("fastest_laps") or 0)
    total_fl = max(total_fl, fl_extra)

    try:
        pts_val = float(s['points'])
        if math.isnan(pts_val):
            pts_display = "0"
        elif pts_val.is_integer():
            pts_display = str(int(pts_val))
        else:
            pts_display = str(round(pts_val, 1))
    except Exception:
        pts_display = "0"

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


def complete_season_summaries(year: int) -> None:
    """Background task: fill session summaries for all completed Race and Sprint
    sessions in the season. Safe to call repeatedly.
    """
    if year < 2018:
        return
    _aggregate_season_stats(year, basic_only=True)
    cache = _SEASON_STATS_CACHE.get(year)
    if not cache:
        return
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as exc:
        _log(f"[complete] WARN: schedule {year} failed: {exc}")
        return
    cols = list(getattr(schedule, "columns", []))
    now_utc = datetime.now(timezone.utc)

    def _session_dt_utc(row, code: str):
        name_cols = [
            c
            for c in cols
            if c.startswith("Session") and c[-1:].isdigit() and not c.endswith("Utc") and not c.endswith("DateUtc")
        ]
        for nc in name_cols:
            idx = nc[len("Session") :]
            dcol = f"Session{idx}DateUtc"
            name = str(row.get(nc, "") or "").strip().lower()
            if code == "R" and (name == "race" or "grand prix" in name or "grandprix" in name):
                return row.get(dcol)
            if code == "S" and (name == "sprint" or ("sprint" in name and "qual" not in name and "shootout" not in name)):
                return row.get(dcol)
        return None

    to_process: list[tuple[int, str]] = []
    for _, row in schedule.iterrows():
        rnd_val = row.get("RoundNumber", None)
        try:
            rnd = int(rnd_val)
        except Exception:
            continue
        for code in ("R", "S"):
            ts = _session_dt_utc(row, code)
            try:
                if pd is not None and isinstance(ts, pd.Timestamp):
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("UTC")
                    ts = ts.tz_convert("UTC").to_pydatetime()
            except Exception:
                pass
            if isinstance(ts, datetime) and ts <= now_utc:
                to_process.append((rnd, code))
    # Deduplicate and skip processed
    processed: set = cache.get("processed_sessions", set())  # type: ignore
    if not isinstance(processed, set):
        processed = set()
    to_process = [(r, c) for (r, c) in to_process if (r, c) not in processed]
    if not to_process:
        return
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
    except Exception:
        ThreadPoolExecutor = None  # type: ignore
    if ThreadPoolExecutor is not None:
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = [ex.submit(_ensure_session_summary, year, r, c) for (r, c) in to_process]
            for f in futs:
                try:
                    f.result(timeout=None)
                except Exception as exc:
                    _log(f"[complete] WARN: summary task failed: {exc}")
    else:
        for (r, c) in to_process:
            _ensure_session_summary(year, r, c)


def constructor_stats(ctx: Context, year: int, team_name: str) -> str:
    """Build a stats summary for a constructor in a given season (2018+)."""
    _suppress_ff1_logs()
    if year < 2018:
        return "Constructor statistics are only available from 2018 onwards."
    _, teams = _aggregate_season_stats(year)
    s = teams.get(team_name)
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


def results(ctx: Context) -> str:
    """Show season results: driver and constructor standings side-by-side.

    Implementation switched to FastF1-only aggregation (no Ergast), summing
    points from Race ('R') and Sprint ('S') sessions across the selected
    season. This avoids issues with Ergast responses in ongoing seasons.

    - Drivers table: Pos, Driver, Nat, Team, Pts (sorted by points desc)
    - Teams table:   Pos, Team, Pts (sorted by points desc)
    - Seasons prior to 2018 are not supported here (no live timing/results).
    """
    _suppress_ff1_logs()
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
    drivers_data, teams_data = _aggregate_season_stats(year, basic_only=True)

    if not drivers_data and not teams_data:
        _log(f"[results] WARN: No data aggregated for {year}")
        return f"No data Available for {year} F1 Season..."

    driver_points: dict[str, float] = {abbr: d["points"] for abbr, d in drivers_data.items()}
    driver_name: dict[str, str] = {abbr: d["name"] for abbr, d in drivers_data.items()}
    driver_nat: dict[str, str] = {abbr: d["nat"] for abbr, d in drivers_data.items()}
    driver_team: dict[str, str] = {abbr: d["team"] for abbr, d in drivers_data.items()}

    team_points: dict[str, float] = {team: t["points"] for team, t in teams_data.items() if "points" in t}
    # Fallback if team points not in teams_data (old logic didn't have it but I'll add it to _aggregate_season_stats)
    if not team_points:
        from collections import defaultdict
        team_points = defaultdict(float)
        for abbr, d in drivers_data.items():
            if d["team"]:
                team_points[d["team"]] += d["points"]

    # Prepare drivers table lines
    left_lines: list[str] = []
    # Compute dynamic widths for driver table
    # Gather values for width computation
    driver_rows_preview = []
    for abbr, pts in driver_points.items():
        driver_rows_preview.append(
            {
                "name": driver_name.get(abbr, abbr),
                "nat": driver_nat.get(abbr, ""),
                "team": driver_team.get(abbr, ""),
                "pts": int(pts) if float(pts).is_integer() else round(pts, 1),
            }
        )

    def _w(values: list[str], header: str, *, min_w: int, max_w: int) -> int:
        base = max(len(header), max((len(str(v)) for v in values), default=0))
        base = max(min_w, base)
        base = min(max_w, base)
        return base

    POSW = _w(
        [str(i) for i in range(1, max(1, len(driver_rows_preview)) + 1)],
        "Pos",
        min_w=2,
        max_w=3,
    )
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
        try:
            fpts = float(pts)
            if math.isnan(fpts):
                pts_display = "0"
            elif fpts.is_integer():
                pts_display = str(int(fpts))
            else:
                pts_display = str(round(fpts, 1))
        except Exception:
            pts_display = "0"
        left_lines.append(
            f"{pos_counter:>{POSW}}  {clip(name, DNAMEW):<{DNAMEW}}  {clip(nat, NATW):<{NATW}}  {clip(team, TNAMEW):<{TNAMEW}}  {pts_display:>{PTSW}}"
        )
        d_count += 1
        pos_counter += 1
    if d_count:
        left_lines.append(left_hr)

    # Prepare teams table lines
    right_lines: list[str] = []
    # Compute dynamic widths for constructor table
    team_rows_preview = []
    for t, p in team_points.items():
        try:
            fp = float(p)
            if math.isnan(fp):
                p_disp = "0"
            elif fp.is_integer():
                p_disp = str(int(fp))
            else:
                p_disp = str(round(fp, 1))
        except Exception:
            p_disp = "0"
        team_rows_preview.append({"team": t, "pts": p_disp})

    T_POSW = _w(
        [str(i) for i in range(1, max(1, len(team_rows_preview)) + 1)],
        "Pos",
        min_w=2,
        max_w=3,
    )
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
        try:
            fpts = float(pts)
            if math.isnan(fpts):
                pts_display = "0"
            elif fpts.is_integer():
                pts_display = str(int(fpts))
            else:
                pts_display = str(round(fpts, 1))
        except Exception:
            pts_display = "0"
        right_lines.append(
            f"{pos_counter:>{T_POSW}}  {clip(team, T_TEAMW):<{T_TEAMW}}  {pts_display:>{T_PTSW}}"
        )
        t_count += 1
        pos_counter += 1
    if t_count:
        right_lines.append(right_hr)

    # No data handling
    if d_count == 0 and t_count == 0:
        _log(f"[results] WARN: No data aggregated for {year}")
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
    _suppress_ff1_logs()
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

    # pandas is optionally available via module-level import as `pd`

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
                    parts = [
                        p
                        for p in name.replace("(", " ").replace(")", " ").split()
                        if p.isalpha()
                    ]
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
                c
                for c in schedule_sorted.columns
                if c.startswith("Session")
                and c[-1:].isdigit()
                and not c.endswith("Utc")
                and not c.endswith("DateUtc")
            ]

            def _find_session_dt(row, matchers):
                chosen_idx = None
                for c in sorted(session_name_cols):
                    name = str(row.get(c, ""))
                    if not name:
                        continue
                    lname = name.lower()
                    if any(m(lname) for m in matchers):
                        idx = c[len("Session") :]
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
                        lambda n: ("qualifying" in n)
                        and ("sprint" not in n)
                        and ("shootout" not in n),
                    ],
                )

            def find_sprint_dt(row):
                return _find_session_dt(
                    row,
                    [
                        lambda n: n == "sprint",
                        lambda n: ("sprint" in n)
                        and ("qual" not in n)
                        and ("shootout" not in n),
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

                rows_data.append(
                    {
                        "rnd": rn,
                        "country": country,
                        "location": location,
                        "sq": sprint_q_str,
                        "s": sprint_str,
                        "q": quali_str,
                        "r": local_str,
                    }
                )
                events_count += 1

            # Compute dynamic widths with sensible caps
            CW = clamp_width(
                [str(r["country"]) for r in rows_data], H_COUNTRY, min_w=10, max_w=25
            )
            LW = clamp_width(
                [str(r["location"]) for r in rows_data], H_LOCATION, min_w=10, max_w=18
            )
            SQW = clamp_width(
                [str(r["sq"]) for r in rows_data], H_SPRINTQ, min_w=5, max_w=22
            )
            SW = clamp_width(
                [str(r["s"]) for r in rows_data], H_SPRINT, min_w=5, max_w=22
            )
            QW = clamp_width(
                [str(r["q"]) for r in rows_data], H_RACEQ, min_w=5, max_w=22
            )
            RW = clamp_width(
                [str(r["r"]) for r in rows_data], H_RACE, min_w=5, max_w=22
            )

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
            or (
                "EventStartDate"
                if "EventStartDate" in schedule_sorted.columns
                else None
            )
        )
        quali_date_col = (
            "QualifyingDate" if "QualifyingDate" in schedule_sorted.columns else None
        ) or find_col_contains("qualifying", "date")

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

            rows_data.append(
                {
                    "rnd": rn,
                    "country": country,
                    "location": location,
                    "sq": sprint_q_str,
                    "s": sprint_str,
                    "q": quali_str,
                    "r": race_str,
                }
            )
            events_count += 1

        # Compute dynamic widths with sensible caps
        CW = clamp_width(
            [str(r["country"]) for r in rows_data], H_COUNTRY, min_w=10, max_w=18
        )
        LW = clamp_width(
            [str(r["location"]) for r in rows_data], H_LOCATION, min_w=10, max_w=18
        )
        SQW = clamp_width(
            [str(r["sq"]) for r in rows_data], H_SPRINTQ, min_w=5, max_w=22
        )
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
