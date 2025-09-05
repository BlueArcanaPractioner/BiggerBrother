from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
import json, os, datetime as dt, re

PLAN_DIR = os.environ.get("PLAN_DIR", "data/plans")

@dataclass
class Block:
    title: str
    start: str  # ISO time
    end: str    # ISO time
    kind: str   # "work"|"meal"|"errand"|"break"|"misc"
    notes: Optional[str] = None

class DayPlanner:
    def __init__(self, start_hour=9, end_hour=22):
        self.start_hour = start_hour
        self.end_hour = end_hour
        os.makedirs(PLAN_DIR, exist_ok=True)

    def plan(self, today_tasks: List[Dict], grocery_items: List[str], meal_blocks: List[Dict]) -> Dict:
        """
        today_tasks: [{"title": "...", "mins": 60, "kind":"work"}]
        grocery_items: ["milk","eggs"]
        meal_blocks: [{"title":"Cook lunch","mins":40},{"title":"Cook dinner","mins":50}]
        """
        # naive greedy packing
        date = dt.date.today()
        cursor = dt.datetime.combine(date, dt.time(self.start_hour, 0))
        end_of_day = dt.datetime.combine(date, dt.time(self.end_hour, 0))
        blocks: List[Block] = []

        def push(title, mins, kind, notes=None):
            nonlocal cursor
            start = cursor
            end = start + dt.timedelta(minutes=mins)
            if end > end_of_day: return
            blocks.append(Block(title=title, start=start.isoformat(), end=end.isoformat(), kind=kind, notes=notes))
            cursor = end

        # breakfast buffer
        push("Morning setup", 15, "misc")
        push("Breakfast", 20, "meal")

        # insert tasks and meal blocks alternating
        items = today_tasks.copy()
        mb = meal_blocks.copy()

        while cursor < end_of_day and (items or mb):
            if items:
                t = items.pop(0)
                push(t["title"], t.get("mins", 50), t.get("kind", "work"))
            if mb:
                m = mb.pop(0)
                push(m.get("title", "Cook"), m.get("mins", 40), "meal")
            # short break
            push("Break", 10, "break")

        # if low items exist, add errand slot near late afternoon
        if grocery_items:
            # place at ~17:00 or after current cursor, whichever is later
            slot = dt.datetime.combine(date, dt.time(17, 0))
            if cursor < slot:
                cursor = slot
            push("Grocery run", min(20 + 5*len(grocery_items), 90), "errand",
                 notes=f"Buy: {', '.join(grocery_items)}")

        plan = {"date": date.isoformat(), "blocks": [asdict(b) for b in blocks]}
        json.dump(plan, open(os.path.join(PLAN_DIR, f"{date}.json"), "w"), indent=2)
        return plan

# very small NL helpers for minutes & simple “for 30m” pattern
DUR_RE = re.compile(r"\bfor\s*(\d+)\s*(m|min|minutes|h|hr|hour|hours)\b", re.I)
def parse_duration_mins(text: str, default=50) -> int:
    m = DUR_RE.search(text)
    if not m: return default
    val, unit = int(m.group(1)), m.group(2).lower()
    return val if unit.startswith("m") else val * 60
