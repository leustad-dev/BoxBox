"""Placeholder menu action handlers for BoxBox TUI.

Each handler receives a context dict (may include season, round, session)
and returns a short status string to display in the TUI. Some handlers are
still stubs; calendar renders a formatted table.
"""
from __future__ import annotations

from typing import Dict, Optional
from datetime import date, datetime, timezone

import fastf1


Context = Dict[str, Optional[object]]


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
    TW, DW = 22, 26  # team width, driver width

    def clip(val: object, width: int) -> str:
        s = "" if val is None else str(val)
        return s if len(s) <= width else s[:width]

    header = f"{'Team':<{TW}}  {'Driver':<{DW}}"
    rule_len = max(len(title), len(header))
    hr = "-" * rule_len

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


def constructors(ctx: Context) -> str:
    return "Constructors selected (stub)"


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

    # Fixed widths to keep alignment; clip long values
    title = f"F1 {year} Calendar"
    CW, LW, RW, SW, SQW, QW = 16, 18, 20, 16, 16, 16

    def clip(val: object, width: int) -> str:
        s = "" if val is None else str(val)
        if len(s) <= width:
            return s
        return s[:width]

    header = (
        f"{'Rnd':>3}  {'Country':<{CW}}  {'Location':<{LW}}  "
        f"{'Sprint Qual':<{SQW}}  {'Sprint':<{SW}}  {'Race Qual':<{QW}}  {'Race':<{RW}}"
    )
    rule_len = max(len(header), len(title))
    hr = "-" * rule_len
    lines = [title, header, hr]

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

            for _, row in schedule_sorted.iterrows():
                rn_val = row.get("RoundNumber", "")
                try:
                    rn = int(rn_val)
                except Exception:
                    rn = rn_val

                country = clip(row.get("Country", ""), CW)
                location = clip(str(row.get("Location", "")) or "-", LW)
                dt_utc = row.get("Session5DateUtc", None)
                local_str = clip(format_local(dt_utc), RW)

                sprint_str = "-"
                sprint_q_str = "-"
                if year >= 2021:
                    s_col = find_sprint_dt(row)
                    if s_col:
                        sprint_str = clip(format_local(row.get(s_col, None)), SW)
                    sq_col = find_sprint_quali_dt(row)
                    if sq_col:
                        sprint_q_str = clip(format_local(row.get(sq_col, None)), SQW)

                q_col = find_quali_dt(row)
                if q_col:
                    quali_str = clip(format_local(row.get(q_col, None)), QW)
                else:
                    quali_str = "-"

                lines.append(
                    f"{rn:>3}  {country:<{CW}}  {location:<{LW}}  {sprint_q_str:<{SQW}}  {sprint_str:<{SW}}  {quali_str:<{QW}}  {local_str:<{RW}}"
                )
                lines.append(hr)
                events_count += 1
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

        for _, row in schedule_sorted.iterrows():
            rn_val = row.get("RoundNumber", "")
            try:
                rn = int(rn_val)
            except Exception:
                rn = rn_val

            country = clip(row.get("Country", ""), CW)
            location = clip(str(row.get("Location", "")) or "-", LW)

            race_dt = row.get(race_date_col, None) if race_date_col else None
            race_str = clip(format_local(race_dt), RW) if race_dt is not None else "-"

            sprint_str = "-"  # not applicable pre-2021
            sprint_q_str = "-"

            quali_dt = row.get(quali_date_col, None) if quali_date_col else None
            quali_str = clip(format_local(quali_dt), QW) if quali_dt is not None else "-"

            lines.append(
                f"{rn:>3}  {country:<{CW}}  {location:<{LW}}  {sprint_q_str:<{SQW}}  {sprint_str:<{SW}}  {quali_str:<{QW}}  {race_str:<{RW}}"
            )
            lines.append(hr)
            events_count += 1
    except Exception as exc:
        return f"Loaded schedule for {year}, but formatting failed: {exc}"

    if events_count == 0:
        return no_data_msg

    return "\n".join(lines)


# A simple registry that TUI can use if needed
MENU_ACTIONS = {
    "Live": live_timing,
    "Drivers": drivers,
    "Constructors": constructors,
    "Sessions": sessions,
    "Calendar": calendar,
    "Settings": settings,
    "Help": help_about,
}
