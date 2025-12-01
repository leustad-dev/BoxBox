"""Placeholder menu action handlers for BoxBox TUI.

Each handler receives a context dict (may include season, round, session)
and returns a short status string to display in the TUI. For now, they are stubs.
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
    return "Drivers selected (stub)"


def constructors(ctx: Context) -> str:
    return "Constructors selected (stub)"


def sessions(ctx: Context) -> str:
    return "Sessions selected (stub)"


def settings(ctx: Context) -> str:
    return "Settings selected (stub)"


def help_about(ctx: Context) -> str:
    return "BoxBox CLI — navigate with ← →, Enter to select, q/ESC to exit"


def calendar(ctx: Context) -> str:
    """Fetch and format the current season calendar using FastF1.

    Returns a multi-line string suitable for the TUI status area.
    """
    try:
        year = int(ctx.get("season") or date.today().year)
    except Exception:
        year = date.today().year

    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as exc:
        return f"Failed to load schedule for {year}: {exc}"

    # Prefer the requested columns; fall back to older formatting if missing
    cols = set(schedule.columns)
    has_required = {"RoundNumber", "Country", "Session5DateUtc"}.issubset(cols)

    if has_required:
        try:
            import pandas as pd  # type: ignore
        except Exception:  # pragma: no cover - pandas comes with fastf1
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
            """Return a short timezone abbreviation.

            - If tz_name has spaces (e.g., "Central Standard Time"), compress to initials ("CST").
            - If tz_name already looks short (no space), return as-is.
            - If tz_name is None/empty, fall back to a numeric offset like UTC+HH:MM.
            """
            try:
                name = (tz_name or "").strip()
                if name:
                    if " " in name:
                        # Compress multi-word names to initials
                        parts = [p for p in name.replace("(", " ").replace(")", " ").split() if p.isalpha()]
                        if parts:
                            return "".join(p[0].upper() for p in parts)
                    # Already an abbreviation
                    return name
                # No tz name -> compute offset
                # Support pandas.Timestamp and datetime
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
            """Format a UTC datetime to local with abbreviated timezone."""
            loc = to_local(dt_obj)
            try:
                if hasattr(loc, "tzname"):
                    tz_name = loc.tzname()
                else:
                    tz_name = None
                abbr = _abbr_tz(tz_name, loc)
                if hasattr(loc, "strftime"):
                    return loc.strftime("%b %d %Y %H:%M ") + abbr
                return str(loc)
            except Exception:
                return str(loc) if loc is not None else ""

        # Build a neat header with explicit column names (include Location)
        title = f"F1 {year} Calendar"
        # Define fixed column widths and a safe clipper to avoid shifting when values are long
        # Country, Location, Race, Sprint, Sprint Qual, Qualifying widths
        CW, LW, RW, SW, SQW, QW = 16, 18, 20, 16, 16, 16

        def clip(val: object, width: int) -> str:
            s = "" if val is None else str(val)
            if len(s) <= width:
                return s
            # Prefer hard clip instead of ellipsis to keep alignment perfectly consistent
            return s[:width]

        header = (
            f"{'Rnd':>3}  {'Country':<{CW}}  {'Location':<{LW}}  "
            f"{'Sprint Qual':<{SQW}}  {'Sprint':<{SW}}  {'Race Qual':<{QW}}  {'Race':<{RW}}"
        )
        rule_len = max(len(header), len(title))
        hr = "-" * rule_len  # ASCII rule for broad terminal compatibility
        lines = [title, header, hr]

        try:
            # Ensure sorted by round number if possible
            try:
                schedule_sorted = schedule.sort_values("RoundNumber")
            except Exception:
                schedule_sorted = schedule

            # Pre-compute which Session column stores specific sessions by scanning names
            session_name_cols = [c for c in schedule_sorted.columns if c.startswith("Session") and c[-1:].isdigit() and not c.endswith("Utc") and not c.endswith("DateUtc")]

            def _find_session_dt(row, matchers):
                """Generic helper to find a session date column by name.
                `matchers` is a list of callables that accept a lowercase session name
                and return True if it matches the desired session.
                """
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
                # Qualifying (exclude sprint shootout)
                return _find_session_dt(
                    row,
                    [
                        lambda n: n == "qualifying",
                        lambda n: ("qualifying" in n) and ("sprint" not in n) and ("shootout" not in n),
                    ],
                )

            def find_sprint_dt(row):
                # Sprint race (not sprint qualifying/shootout)
                return _find_session_dt(
                    row,
                    [
                        lambda n: n == "sprint",
                        lambda n: ("sprint" in n) and ("qual" not in n) and ("shootout" not in n),
                    ],
                )

            def find_sprint_quali_dt(row):
                # Sprint Qualifying (older naming) or Sprint Shootout (newer naming)
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

                # Sprint time (if available)
                sprint_col = find_sprint_dt(row)
                if sprint_col:
                    s_dt_utc = row.get(sprint_col, None)
                    sprint_str = clip(format_local(s_dt_utc), SW)
                else:
                    sprint_str = "-"

                # Sprint Qualifying / Shootout time (if available)
                sprint_q_col = find_sprint_quali_dt(row)
                if sprint_q_col:
                    sq_dt_utc = row.get(sprint_q_col, None)
                    sprint_q_str = clip(format_local(sq_dt_utc), SQW)
                else:
                    sprint_q_str = "-"

                # Qualifying time (if available)
                quali_col = find_quali_dt(row)
                if quali_col:
                    q_dt_utc = row.get(quali_col, None)
                    quali_str = clip(format_local(q_dt_utc), QW)
                else:
                    quali_str = "-"

                lines.append(
                    f"{rn:>3}  {country:<{CW}}  {location:<{LW}}  {sprint_q_str:<{SQW}}  {sprint_str:<{SW}}  {quali_str:<{QW}}  {local_str:<{RW}}"
                )
                # Row separator for table-like readability
                lines.append(hr)
        except Exception as exc:
            return f"Loaded schedule for {year}, but formatting failed: {exc}"

        return "\n".join(lines)

    # Fallback to previous, more permissive formatting
    use_event = "EventName" if "EventName" in cols else ("OfficialEventName" if "OfficialEventName" in cols else None)
    use_country = "Country" if "Country" in cols else None
    use_round = "RoundNumber" if "RoundNumber" in cols else None
    use_date = "EventDate" if "EventDate" in cols else ("EventStartDate" if "EventStartDate" in cols else None)

    lines = [f"F1 {year} Calendar"]
    try:
        for _, row in schedule.iterrows():
            rn = str(row[use_round]) if use_round else ""
            name = str(row[use_event]) if use_event else ""
            country = str(row[use_country]) if use_country else ""
            dt = row[use_date] if use_date else None
            try:
                dts = dt.strftime("%b %d") if hasattr(dt, "strftime") else (str(dt) if dt is not None else "")
            except Exception:
                dts = str(dt) if dt is not None else ""

            label = name or country
            prefix = f"{rn:>2}. " if rn else " - "
            suffix = f"  ({dts})" if dts else ""
            if country and name and country not in name:
                label = f"{country} — {name}"
            elif not label:
                label = "Event"
            lines.append(f"{prefix}{label}{suffix}")
    except Exception as exc:
        return f"Loaded schedule for {year}, but formatting failed: {exc}"

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
