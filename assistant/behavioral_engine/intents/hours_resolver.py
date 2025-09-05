from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, time, date
from zoneinfo import ZoneInfo
import json, os

LOCAL_TZ = ZoneInfo(os.environ.get("LOCAL_TZ", "America/Kentucky/Louisville"))
HOURS_DB = os.environ.get("HOURS_DB", "data/hours_directory.json")

# Minimal structure:
# {
#   "chiropractor": {"tz":"America/Kentucky/Louisville",
#     "hours":{"MO":[["09:00","17:00"]], "TU":[["09:00","17:00"]], "WE":[["09:00","17:00"]],
#              "TH":[["09:00","17:00"]], "FR":[["09:00","17:00"]], "SA":[], "SU":[]}}
# }

DAYS = ["MO","TU","WE","TH","FR","SA","SU"]

@dataclass
class OpenHours:
    tz: str
    windows: Dict[str, List[Tuple[str,str]]]  # daycode -> list of [("09:00","17:00"), ...]

def _load_hours() -> Dict[str, OpenHours]:
    os.makedirs(os.path.dirname(HOURS_DB), exist_ok=True)
    if not os.path.exists(HOURS_DB):
        json.dump({}, open(HOURS_DB, "w"), indent=2)
    raw = json.load(open(HOURS_DB))
    out = {}
    for k,v in raw.items():
        out[k.lower()] = OpenHours(tz=v.get("tz", "UTC"), windows=v.get("hours", {}))
    return out

def next_open_slot(who: str, not_after: Optional[time] = None, weekday_only: bool = False,
                   earliest: Optional[datetime] = None, min_duration_minutes: int = 30) -> Optional[datetime]:
    book = _load_hours()
    who_key = who.lower().strip()
    rec = book.get(who_key)  # None means unknown—fall back to weekday 9–5
    tz = ZoneInfo(rec.tz) if rec else LOCAL_TZ

    def windows_for(dcode: str) -> List[Tuple[time,time]]:
        if rec:
            wins = []
            for s,e in rec.windows.get(dcode, []):
                hs, ms = map(int, s.split(":")); he, me = map(int, e.split(":"))
                wins.append((time(hs, ms), time(he, me)))
            return wins
        # default fallback: MO–FR 09:00–17:00
        return [(time(9,0), time(17,0))] if dcode in DAYS[:5] else []

    now = (earliest or datetime.now(tz)).astimezone(tz)
    start_date = now.date()
    for delta in range(0, 10):  # search the next 10 days
        d = start_date + timedelta(days=delta)
        dcode = DAYS[d.weekday()]
        if weekday_only and dcode in ("SA","SU"):
            continue
        for ws, we in windows_for(dcode):
            if not_after:
                we = min(we, not_after)
            candidate = datetime.combine(d, max(ws, now.time()) if d==start_date else ws, tzinfo=tz)
            end_time = (datetime.combine(d, we, tzinfo=tz))
            # ensure window is long enough
            if candidate + timedelta(minutes=min_duration_minutes) <= end_time:
                if candidate.time() < ws: candidate = datetime.combine(d, ws, tzinfo=tz)
                if candidate <= now:  # past; push to next feasible minute
                    candidate = now + timedelta(minutes=5)
                if candidate + timedelta(minutes=min_duration_minutes) <= end_time:
                    return candidate.astimezone(LOCAL_TZ)
    return None  # nothing within 10 days
