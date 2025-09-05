from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Literal
import re

Channel = Literal["chat", "checkin", "email"]

@dataclass
class Intent:
    kind: Literal["schedule_appointment", "symptom_note", "todo",
                  "plan_today", "cook", "pantry_add", "pantry_use"]
    slots: Dict[str, str]
    raw_text: str
    channel: Channel

_RX_APPT = re.compile(
    r"\b(schedule|book|make)\s+(an?\s+)?(appointment|appt)\s+(with|at)\s+(my\s+)?(?P<who>[a-z\s]+)\b",
    re.I
)
_RX_SYMPTOM = re.compile(
    r"\b(my|the)\s+(?P<bodypart>neck|back|knee|shoulder|head)\s+(is|started|starting)\s+to\s+hurt\b",
    re.I
)
_RX_RULES = re.compile(
    r"\bnot\s+(after|past)\s+(?P<hour>\d{1,2})(:(?P<min>\d{2}))?\s*(?P<ampm>am|pm)\b|\bno\s+weekend(s)?\b",
    re.I
)

def parse_intent(text: str, channel: Channel) -> Optional[Intent]:
    t = text.strip()

    # Appointment intent (e.g., “schedule an appointment with my chiropractor”)
    if _RX_APPT.search(t) or "chiropractor" in t.lower():
        who = "chiropractor"
        m = _RX_APPT.search(t)
        if m and m.group("who"):
            who = m.group("who").strip()

        constraints = []
        for m in _RX_RULES.finditer(t):
            gd = m.groupdict()
            if gd.get("hour"):
                hh = int(gd["hour"])
                mm = int(gd.get("min") or 0)
                ampm = (gd.get("ampm") or "").lower()
                if ampm == "pm" and hh != 12:
                    hh += 12
                if ampm == "am" and hh == 12:
                    hh = 0
                constraints.append(f"not_after:{hh:02d}:{mm:02d}")
            if "weekend" in m.group(0).lower():
                constraints.append("weekday_only:true")

        return Intent(
            kind="schedule_appointment",
            slots={"who": who, "constraints": ";".join(constraints)},
            raw_text=t,
            channel=channel,
        )

    # Symptom note
    m = _RX_SYMPTOM.search(t)
    if m:
        return Intent(
            kind="symptom_note",
            slots={"bodypart": m.group("bodypart").lower()},
            raw_text=t,
            channel=channel,
        )

    # Quick command shims (forwarded to other subsystems)
    low = t.lower()
    if low.startswith("plan today:"):
        return Intent("plan_today", {"payload": t.split(":", 1)[1].strip()}, t, channel)
    if low.startswith(("cook:", "cooking:")):
        return Intent("cook", {"payload": t.split(":", 1)[1].strip()}, t, channel)
    if low.startswith("pantry add "):
        return Intent("pantry_add", {"payload": t[11:].strip()}, t, channel)
    if low.startswith("pantry use "):
        return Intent("pantry_use", {"payload": t[11:].strip()}, t, channel)
    if low.startswith(("todo:", "add todo:")) or low.startswith("remind me to "):
        payload = t.split(":", 1)[-1].strip() if ":" in t else t
        return Intent("todo", {"text": payload}, t, channel)

    return None
