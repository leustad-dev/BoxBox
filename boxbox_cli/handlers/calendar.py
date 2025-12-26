from __future__ import annotations

import fastf1
from datetime import date, datetime, timezone
from typing import Dict, Optional
from ..utils.logger import _suppress_ff1_logs, _log_exception

try:
    import pandas as pd
except Exception:
    pd = None

Context = Dict[str, Optional[object]]

def calendar(ctx: Context) -> str:
    """Fetch and format the season calendar using FastF1."""
    _suppress_ff1_logs()
    try:
        year = int(ctx.get("season") or date.today().year)
    except Exception:
        year = date.today().year

    try:
        backend = None if year >= 2018 else "ergast"
        schedule = fastf1.get_event_schedule(year, backend=backend)
    except Exception as exc:
        _log_exception(f"calendar() - failed to load schedule for {year}", exc)
        return f"Failed to load schedule for {year}: {exc}"

    no_data_msg = f"No data Available for {year} F1 Season..."
    if schedule is None or (hasattr(schedule, "empty") and schedule.empty):
        return no_data_msg

    def to_local(ts: datetime) -> datetime:
        """Convert a UTC timestamp to the user's local timezone."""
        try:
            local_tz = datetime.now().astimezone().tzinfo
            if pd is not None and isinstance(ts, pd.Timestamp):
                if ts.tzinfo is None: ts = ts.tz_localize("UTC")
                return ts.tz_convert(local_tz)
            if isinstance(ts, datetime):
                if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
                return ts.astimezone(local_tz)
        except Exception: pass
        return ts

    def _abbr_tz(tz_name: Optional[str], dt_obj: datetime) -> str:
        """Get an abbreviated timezone name (e.g., 'EST' or 'UTC+05:00')."""
        try:
            name = (tz_name or "").strip()
            if name:
                if " " in name:
                    parts = [p for p in name.replace("(", " ").replace(")", " ").split() if p.isalpha()]
                    if parts: return "".join(p[0].upper() for p in parts)
                return name
            if hasattr(dt_obj, "utcoffset") and dt_obj.utcoffset() is not None:
                offset = dt_obj.utcoffset()
                total_minutes = abs(int(offset.total_seconds() // 60))
                sign = "+" if offset.total_seconds() >= 0 else "-"
                return f"UTC{sign}{total_minutes // 60:02d}:{total_minutes % 60:02d}"
        except Exception: pass
        return "UTC"

    def format_local(dt_obj: datetime) -> str:
        """Format a timestamp into a human-readable local date and time string."""
        loc = to_local(dt_obj)
        try:
            abbr = _abbr_tz(loc.tzname() if hasattr(loc, "tzname") else None, loc)
            return loc.strftime("%b %d %Y %H:%M ") + abbr if hasattr(loc, "strftime") else str(loc)
        except Exception: return str(loc) if loc is not None else ""

    title = f"F1 {year} Calendar"

    def clip(val: object, width: int) -> str:
        """Clip a string to a specified width."""
        s = "" if val is None else str(val)
        return s if len(s) <= width else s[:width]

    def clamp_width(values: list[str], header: str, *, min_w: int, max_w: int) -> int:
        """Calculate optimal column width based on content and header constraints."""
        base = max(len(header), max((len(v) for v in values), default=0))
        return max(min_w, min(max_w, base))

    cols = set(schedule.columns)
    modern = {"RoundNumber", "Country", "Session5DateUtc"}.issubset(cols)

    if modern:
        try:
            schedule_sorted = schedule.sort_values("RoundNumber")
        except Exception as exc:
            _log_exception(f"calendar() - failed to sort schedule for {year}", exc)
            schedule_sorted = schedule

        session_name_cols = [c for c in schedule_sorted.columns if c.startswith("Session") and c[-1:].isdigit() and not c.endswith("Utc") and not c.endswith("DateUtc")]

        def _find_session_dt(row, matchers):
            for nc in session_name_cols:
                val = str(row.get(nc, "") or "").lower()
                if any(m in val for m in matchers):
                    idx = nc[len("Session") :]
                    return row.get(f"Session{idx}DateUtc")
            return None

        find_quali_dt = lambda r: _find_session_dt(r, ["qualifying"])
        find_sprint_dt = lambda r: _find_session_dt(r, ["sprint"]) if not any("qual" in str(r.get(nc, "")).lower() for nc in session_name_cols if "sprint" in str(r.get(nc, "")).lower()) else None

        # Fixing find_sprint_dt logic to match original more closely if needed,
        # but the original had some complex logic. Let's simplify.
        def find_sprint_dt_full(row):
            for nc in session_name_cols:
                val = str(row.get(nc, "") or "").lower()
                if "sprint" in val and "qual" not in val and "shootout" not in val:
                    idx = nc[len("Session") :]
                    return row.get(f"Session{idx}DateUtc")
            return None

        # Redefining helpers properly
        def _get_dt(row, matchers, excluders=None):
            for nc in session_name_cols:
                val = str(row.get(nc, "") or "").lower()
                if any(m in val for m in matchers):
                    if excluders and any(e in val for e in excluders): continue
                    idx = nc[len("Session") :]
                    return row.get(f"Session{idx}DateUtc")
            return None

        find_quali_dt = lambda r: _get_dt(r, ["qualifying"], ["sprint"])
        find_sprint_dt = lambda r: _get_dt(r, ["sprint"], ["qualifying", "shootout"])
        find_sprint_quali_dt = lambda r: _get_dt(r, ["sprint qualifying", "sprint shootout"])

        rows_data = []
        for _, row in schedule_sorted.iterrows():
            rows_data.append({
                "Rnd": str(row.get("RoundNumber", "")),
                "Country": str(row.get("Country", "")),
                "Location": str(row.get("Location", "")),
                "SQ": format_local(find_sprint_quali_dt(row)),
                "S": format_local(find_sprint_dt(row)),
                "Q": format_local(find_quali_dt(row)),
                "R": format_local(row.get("Session5DateUtc")),
            })

        show_sprint = any(r["SQ"] or r["S"] for r in rows_data)

        W_RND = clamp_width([r["Rnd"] for r in rows_data], "Rnd", min_w=3, max_w=3)
        W_COUNTRY = clamp_width([r["Country"] for r in rows_data], "Country", min_w=10, max_w=20)
        W_LOC = clamp_width([r["Location"] for r in rows_data], "Location", min_w=10, max_w=20)
        W_SQ = clamp_width([r["SQ"] for r in rows_data], "Sprint Qual", min_w=15, max_w=24) if show_sprint else 0
        W_S = clamp_width([r["S"] for r in rows_data], "Sprint", min_w=15, max_w=24) if show_sprint else 0
        W_Q = clamp_width([r["Q"] for r in rows_data], "Race Qual", min_w=15, max_w=24)
        W_R = clamp_width([r["R"] for r in rows_data], "Race", min_w=15, max_w=24)

        header = f"{'Rnd':>{W_RND}}  {'Country':<{W_COUNTRY}}  {'Location':<{W_LOC}}"
        if show_sprint: header += f"  {'Sprint Qual':<{W_SQ}}  {'Sprint':<{W_S}}"
        header += f"  {'Race Qual':<{W_Q}}  {'Race':<{W_R}}"

        lines = [title, "", header, "-" * len(header)]
        for r in rows_data:
            line = f"{r['Rnd']:>{W_RND}}  {clip(r['Country'], W_COUNTRY):<{W_COUNTRY}}  {clip(r['Location'], W_LOC):<{W_LOC}}"
            if show_sprint: line += f"  {r['SQ']:<{W_SQ}}  {r['S']:<{W_S}}"
            line += f"  {r['Q']:<{W_Q}}  {r['R']:<{W_R}}"
            lines.append(line)
        return "\n".join(lines)
    else:
        # Legacy/Ergast
        rows_data = []
        for _, row in schedule.iterrows():
            rows_data.append({
                "Rnd": str(row.get("RoundNumber", "")),
                "Country": str(row.get("Country", "")),
                "Location": str(row.get("Location", "")),
                "Date": format_local(row.get("EventDate")),
            })
        W_RND = clamp_width([r["Rnd"] for r in rows_data], "Rnd", min_w=3, max_w=3)
        W_COUNTRY = clamp_width([r["Country"] for r in rows_data], "Country", min_w=12, max_w=20)
        W_LOC = clamp_width([r["Location"] for r in rows_data], "Location", min_w=12, max_w=20)
        W_DATE = clamp_width([r["Date"] for r in rows_data], "Event Date", min_w=15, max_w=30)

        header = f"{'Rnd':>{W_RND}}  {'Country':<{W_COUNTRY}}  {'Location':<{W_LOC}}  {'Event Date':<{W_DATE}}"
        lines = [title, "", header, "-" * len(header)]
        for r in rows_data:
            lines.append(f"{r['Rnd']:>{W_RND}}  {clip(r['Country'], W_COUNTRY):<{W_COUNTRY}}  {clip(r['Location'], W_LOC):<{W_LOC}}  {r['Date']:<{W_DATE}}")
        return "\n".join(lines)
