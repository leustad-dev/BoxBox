from __future__ import annotations

import json
import os
import math
import fastf1
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from collections import defaultdict
from .logger import _suppress_ff1_logs, _log, _log_exception

try:
    import pandas as pd
except Exception:
    pd = None

_SEASON_STATS_CACHE: dict[int, dict] = {}

def _get_cache_file_path(year: int) -> str:
    """Get the path to the season stats cache file."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cache_dir = os.path.join(project_root, "cache", "stats")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"stats_{year}.json")

def _load_stats_from_disk(year: int) -> Optional[dict]:
    """Load season stats from disk if available and still valid."""
    path = _get_cache_file_path(year)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Check if the cache is potentially outdated by comparing the last event date
        # If the year is in the past, the cache is likely fine.
        # For the current year, we might want to check more frequently.
        # For now, we'll implement a simple "last_updated" check.
        # But a better way is to check the schedule.
        return data
    except Exception as exc:
        _log_exception(f"[stats] Failed to load disk cache for {year}", exc)
        return None

def _save_stats_to_disk(year: int, data: dict) -> None:
    """Save season stats to disk."""
    path = _get_cache_file_path(year)
    try:
        # Prepare for JSON (convert sets to lists)
        json_data = {
            "drivers": {k: v for k, v in data["drivers"].items()},
            "teams": {k: v for k, v in data["teams"].items()},
            "basic_ready": data["basic_ready"],
            "processed_sessions": [list(item) for item in data.get("processed_sessions", [])],
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        _log_exception(f"[stats] Failed to save disk cache for {year}", exc)

def _aggregate_season_stats(year: int, *, basic_only: bool = True):
    """Aggregate season-wide stats for drivers and teams (2018+)."""
    _suppress_ff1_logs()
    cached = _SEASON_STATS_CACHE.get(year)
    if cached and cached.get("basic_ready"):
        return cached["drivers"], cached["teams"]

    # Try loading from disk
    disk_data = _load_stats_from_disk(year)
    if disk_data:
        # Check if we should re-aggregate (e.g. current year and last update > 1 day ago)
        # For simplicity in this first step, if it's on disk we use it.
        # We can refine the invalidation logic later.
        
        # Convert list of lists back to set of tuples
        processed = set()
        for item in disk_data.get("processed_sessions", []):
            processed.add(tuple(item))
            
        _SEASON_STATS_CACHE[year] = {
            "drivers": disk_data["drivers"],
            "teams": disk_data["teams"],
            "basic_ready": disk_data["basic_ready"],
            "enriched": set(),
            "session_summaries": {},
            "processed_sessions": processed,
        }
        return disk_data["drivers"], disk_data["teams"]

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
        _log_exception(f"[stats] ERROR loading schedule {year}", exc)
        return drivers, teams

    cols = list(getattr(schedule, "columns", []))
    now_utc = datetime.now(timezone.utc)

    def _session_dt_utc(row, code: str):
        name_cols = [
            c for c in cols
            if c.startswith("Session") and c[-1:].isdigit() and not c.endswith("Utc") and not c.endswith("DateUtc")
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
                _log_exception(f"[stats] WARN: qual rnd={rnd} failed", exc)

        to_consider = ["R"]
        fmt = str(row.get("EventFormat", "") or "").lower()
        if "sprint" in fmt:
            to_consider.append("S")
        else:
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
                s.load(telemetry=False, laps=not basic_only, weather=False, messages=False)
                dfr = getattr(s, "results", None)
                if dfr is None or getattr(dfr, "empty", False):
                    continue
                fl_abbr = ""
                if not basic_only:
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
                    if name: d["name"] = name
                    if team: d["team"] = team
                    if nat: d["nat"] = d["nat"] or nat
                    if num and not d["number"]: d["number"] = str(num)

                    try:
                        meta = s.get_driver(abbr) or {}
                        if not nat:
                            nat = meta.get("Nationality") or meta.get("CountryCode") or meta.get("Country") or ""
                        if not d["nat"] and nat: d["nat"] = nat
                        if not d["dob"]:
                            d["dob"] = meta.get("DateOfBirth") or meta.get("DOB") or meta.get("BirthDate")
                    except Exception:
                        pass

                    try:
                        if math.isnan(float(pos)): ipos = 0
                        else: ipos = int(pos)
                        if ipos == 1: d["wins"] += 1
                        if 1 <= ipos <= 3: d["podiums"] += 1
                        if ipos > 0:
                            d["avg_finish_sum"] += ipos
                            d["avg_finish_cnt"] += 1
                    except Exception:
                        pass

                    try:
                        if math.isnan(float(grid)): igrid = 0
                        else: igrid = int(grid)
                        if igrid > 0:
                            d["avg_grid_sum"] += igrid
                            d["avg_grid_cnt"] += 1
                    except Exception:
                        pass

                    d["points"] += pts
                    if team: teams[team]["points"] += pts

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
                    tname = drivers[fl_abbr]["team"]
                    if tname:
                        teams[tname]["fastest_laps"] += 1
            except Exception as exc:
                _log(f"[stats] WARN: load rnd={rnd} code={code} failed: {exc}")

    _SEASON_STATS_CACHE[year] = {
        "drivers": drivers,
        "teams": teams,
        "basic_ready": True,
        "enriched": set(),
        "session_summaries": {},
        "processed_sessions": set(),
    }
    _save_stats_to_disk(year, _SEASON_STATS_CACHE[year])
    return drivers, teams

def _ensure_session_summary(year: int, rnd: int, code: str) -> Optional[dict]:
    """Ensure session results are loaded and cached in a summary format."""
    _suppress_ff1_logs()
    cache = _SEASON_STATS_CACHE.get(year)
    if not cache:
        _aggregate_season_stats(year, basic_only=True)
        cache = _SEASON_STATS_CACHE.get(year)
    if not cache: return None
    summaries: dict = cache.setdefault("session_summaries", {})
    key = (int(rnd), str(code))
    if key in summaries: return summaries[key]
    try:
        s = fastf1.get_session(year, rnd, code)
        s.load(telemetry=False, laps=True, weather=False, messages=False)
        laps = getattr(s, "laps", None)
        if laps is None or laps.empty:
            res = {"pitstops": {}, "fastest": ""}
            summaries[key] = res
            return res
        try:
            pit_series = laps["PitInTime"].notna()
            pit_counts = laps.loc[pit_series, ["Driver", "PitInTime"]].groupby("Driver").size()
            pit_map = {str(k): int(v) for k, v in pit_counts.to_dict().items()}
        except Exception:
            pit_map = {}
        fastest = ""
        try:
            fl = laps.pick_fastest()
            fastest = str(fl.get("Driver", "") or "").strip()
        except Exception:
            fastest = ""
        res = {"pitstops": pit_map, "fastest": fastest}
        summaries[key] = res
        processed: set = cache.setdefault("processed_sessions", set())
        processed.add(key)
        return res
    except Exception as exc:
        _log_exception(f"[summary] WARN: load summary rnd={rnd} code={code} failed", exc)
        return None

def _enrich_driver(year: int, abbr: str) -> None:
    """Add detailed stats (pitstops, fastest laps) to a driver's record for a given year."""
    cache = _SEASON_STATS_CACHE.get(year)
    if not cache:
        _aggregate_season_stats(year, basic_only=True)
        cache = _SEASON_STATS_CACHE.get(year) or {}
    drivers = cache.get("drivers", {})
    enriched: set = cache.get("enriched", set())
    if abbr in enriched: return
    summaries: dict = cache.get("session_summaries", {})
    pitstops = 0
    fastest_laps = 0
    for data in summaries.values():
        try: pitstops += int(data.get("pitstops", {}).get(abbr, 0))
        except Exception: pass
        try:
            if str(data.get("fastest", "")) == abbr: fastest_laps += 1
        except Exception: pass
    d = drivers.get(abbr)
    if d is not None:
        if pitstops: d["pitstops"] = pitstops
        if fastest_laps: d["fastest_laps"] = fastest_laps
    enriched.add(abbr)

def prewarm_driver_enrichment(year: int, abbrs: list[str]) -> None:
    """Asynchronously fetch and cache session summaries for the most recent races in a season."""
    _suppress_ff1_logs()
    if year < 2018: return
    _aggregate_season_stats(year, basic_only=True)
    cache = _SEASON_STATS_CACHE.get(year)
    if not cache: return
    drivers = cache.get("drivers", {})
    processed = cache.setdefault("processed_sessions", set())

    target_abbrs = {a for a in abbrs if isinstance(a, str) and a}
    if not target_abbrs: return

    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as exc:
        _log_exception(f"[prewarm] WARN: schedule {year} failed", exc)
        return

    cols = list(getattr(schedule, "columns", []))
    now_utc = datetime.now(timezone.utc)

    def _session_dt_utc(row, code: str):
        name_cols = [c for c in cols if c.startswith("Session") and c[-1:].isdigit() and not c.endswith("Utc") and not c.endswith("DateUtc")]
        for nc in name_cols:
            idx = nc[len("Session") :]
            dcol = f"Session{idx}DateUtc"
            name = str(row.get(nc, "") or "").strip().lower()
            if code == "R" and (name == "race" or "grand prix" in name or "grandprix" in name): return row.get(dcol)
            if code == "S" and (name == "sprint" or ("sprint" in name and "qual" not in name and "shootout" not in name)): return row.get(dcol)
        return None

    subset = []
    for _, row in schedule.iterrows():
        rnd_val = row.get("RoundNumber", None)
        try: rnd = int(rnd_val)
        except Exception: continue
        for code in ("R", "S"):
            ts = _session_dt_utc(row, code)
            try:
                if pd is not None and isinstance(ts, pd.Timestamp):
                    if ts.tzinfo is None: ts = ts.tz_localize("UTC")
                    ts = ts.tz_convert("UTC").to_pydatetime()
            except Exception: pass
            if isinstance(ts, datetime) and ts <= now_utc:
                subset.append((rnd, code, ts))

    subset.sort(key=lambda x: x[2], reverse=True)
    subset = subset[:5]

    for (rnd, code, _) in subset:
        if (rnd, code) not in processed:
            _ensure_session_summary(year, rnd, code)

    summaries: dict = cache.get("session_summaries", {})
    for (rnd, code, _) in subset:
        data = summaries.get((rnd, code), {})
        pit_map = data.get("pitstops", {}) or {}
        fl = str(data.get("fastest", "") or "")
        for ab in target_abbrs:
            try: drivers[ab]["pitstops"] += int(pit_map.get(ab, 0))
            except Exception: pass
        if fl:
            try: drivers[fl]["fastest_laps"] += 1
            except Exception: pass
        processed.add((rnd, code))

def complete_season_summaries(year: int) -> None:
    """Asynchronously load all remaining session summaries for an entire season."""
    if year < 2018: return
    _aggregate_season_stats(year, basic_only=True)
    cache = _SEASON_STATS_CACHE.get(year)
    if not cache: return
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as exc:
        _log_exception(f"[complete] WARN: schedule {year} failed", exc)
        return
    cols = list(getattr(schedule, "columns", []))
    now_utc = datetime.now(timezone.utc)

    def _session_dt_utc(row, code: str):
        name_cols = [c for c in cols if c.startswith("Session") and c[-1:].isdigit() and not c.endswith("Utc") and not c.endswith("DateUtc")]
        for nc in name_cols:
            idx = nc[len("Session") :]
            dcol = f"Session{idx}DateUtc"
            name = str(row.get(nc, "") or "").strip().lower()
            if code == "R" and (name == "race" or "grand prix" in name or "grandprix" in name): return row.get(dcol)
            if code == "S" and (name == "sprint" or ("sprint" in name and "qual" not in name and "shootout" in name)): return row.get(dcol)
        return None

    to_process = []
    for _, row in schedule.iterrows():
        rnd_val = row.get("RoundNumber", None)
        try: rnd = int(rnd_val)
        except Exception: continue
        for code in ("R", "S"):
            ts = _session_dt_utc(row, code)
            try:
                if pd is not None and isinstance(ts, pd.Timestamp):
                    if ts.tzinfo is None: ts = ts.tz_localize("UTC")
                    ts = ts.tz_convert("UTC").to_pydatetime()
            except Exception: pass
            if isinstance(ts, datetime) and ts <= now_utc:
                to_process.append((rnd, code))

    processed: set = cache.get("processed_sessions", set())
    to_process = [(r, c) for (r, c) in to_process if (r, c) not in processed]
    if not to_process: return

    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = [ex.submit(_ensure_session_summary, year, r, c) for (r, c) in to_process]
            for f in futs:
                try: f.result()
                except Exception as exc: _log(f"[complete] WARN: summary task failed: {exc}")
    except Exception:
        for (r, c) in to_process:
            _ensure_session_summary(year, r, c)
