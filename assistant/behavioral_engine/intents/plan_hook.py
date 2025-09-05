from __future__ import annotations
from typing import List, Dict
import datetime as dt
from .todo_registry import TodoRegistry
from assistant.home.day_planner import DayPlanner  # re-use your earlier planner
from assistant.home.pantry_manager import PantryManager

def build_today_plan(base_tasks: List[Dict], meal_blocks: List[Dict]) -> Dict:
    """Merge persistent todos (due or undated) + pantry-low errands into today and return a plan."""
    today = dt.date.today().isoformat()
    todos = TodoRegistry().pending_for(today)
    pantry_low = [i.name for i in PantryManager().low_items()]

    # Convert todos into task blocks (lightweight heuristic)
    todo_blocks = [{"title": f"[TODO] {t.text}", "mins": max(15, min(60, t.eta_minutes)), "kind": "work"} for t in todos]

    # Merge: todos first thing, then base tasks, with meals interleaved
    merged = todo_blocks + base_tasks
    plan = DayPlanner().plan(merged, pantry_low, meal_blocks)

    return {"plan": plan, "todos": [t.text for t in todos], "pantry_low": pantry_low}
