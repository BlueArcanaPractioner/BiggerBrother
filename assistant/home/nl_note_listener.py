from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, Optional, Literal

# Unified intent record your system can pass downstream
@dataclass
class Intent:
    kind: Literal[
        "schedule_appointment", "todo", "symptom_note",
        "pantry_add", "pantry_use", "plan_today", "cook"
    ]
    slots: Dict[str, str]                   # extracted entities (e.g., {"who":"chiropractor"})
    raw_text: str
    channel: Literal["chat","checkin","email"]

# ---- Patterns (extend at will) ----
RX_APPT = re.compile(
    r"\b(schedule|book|make)\s+(an?\s+)?(appointment|appt)\s+(with|at)\s+(my\s+)?(?P<who>[a-z\s]+)\b",
    re.I
)
RX_SYMPTOM = re.compile(r"\b(my|the)\s+(?P<bodypart>neck|back|knee|shoulder)\s+(is|started|starting)\s+to\s+hurt\b", re.I)
RX_TIME_RULES = re.compile(r"\bnot\s+(after|past)\s+(?P<hour>\d{1,2})(:?(?P<min>\d{2}))?\s*(am|pm)\b|\bno\s+weekend(s)?\b", re.I)
RX_WEEKDAY_ONLY = re.compile(r"\b(Mon(day)?|Tue(s(day)?)?|Wed(nesday)?|Thu(rs(day)?)?|Fri(day)?)\b", re.I)
RX_CLOSED_HINT = re.compile(r"\b(7\s*pm|weekend)\b.*\b(close|closed)\b", re.I)

def parse_intent(text: str, channel: Intent.__annotations__['channel']) -> Optional[Intent]:
    t = text.strip()

    # Chiropractor / professional appointment example
    if RX_APPT.search(t) or "chiropractor" in t.lower():
        who = "chiropractor"
        m = RX_APPT.search(t)
        if m and m.group("who"):
            who = m.group("who").strip()
        constraints = []
        if RX_CLOSED_HINT.search(t): constraints.append("respect_hours:true")
        if RX_WEEKDAY_ONLY.search(t): constraints.append("weekday_only:true")
        for m in RX_TIME_RULES.finditer(t):
            if m.group("hour"):
                constraints.append(f"not_after:{m.group('hour')}:{m.group('min') or '00'}")
            if "weekend" in m.group(0).lower():
                constraints.append("weekday_only:true")
        return Intent(kind="schedule_appointment", slots={"who": who, "constraints": ";".join(constraints)}, raw_text=t, channel=channel)

    # Symptom logging (so your logger can file it)
