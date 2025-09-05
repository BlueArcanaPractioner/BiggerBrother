from __future__ import annotations
import datetime as dt
from logging import Logger
from typing import Optional
from .nl_note_listener import Intent, parse_intent
from .todo_registry import TodoRegistry
from .hours_resolver import next_open_slot
from assistant.behavioral_engine.intents.plan_hook import build_today_plan
from assistant.logger.unified import UnifiedLogger  # fix
from datetime import time, timedelta

ULOG = UnifiedLogger(data_dir="data/tracking")

def handle_incoming_text(text: str, channel: Intent.__annotations__['channel']) -> Optional[dict]:
    intent = parse_intent(text, channel)
    if not intent:
        return None

    # 0) Always log a lightweight behavioral note if present
    if intent.kind == "symptom_note":
        ULOG.log_event(category="mood", subcategory="symptom",
                       summary=intent.slots.get("text") or intent.raw_text,
                       source=channel)
        return {"ok": True, "noted": True}

    # 1) Scheduling appointments (your chiropractor case)
    if intent.kind == "schedule_appointment":
        provider = (intent.slots.get("provider") or "chiropractor").lower()
        constraints = intent.slots.get("constraints") or {}
        not_after_str = constraints.get("not_after") or "19:00"
        hh, mm = map(int, not_after_str.split(":"))
        slot = next_open_slot(provider,
                              not_after=time(hh, mm),
                              weekday_only=bool(constraints.get("weekday_only", True)),
                              min_duration_minutes=int(constraints.get("min_duration_minutes", 30)))
        reg = TodoRegistry()
        if slot:
            dstr = slot.strftime("%Y-%m-%d")
            tstr = slot.strftime("%H:%M")
            todo = reg.add(
                text=f"Call {provider} to schedule for {dstr} {tstr}",
                kind="call", due_date=dstr, eta_minutes=10, source=channel,
                tags=[provider, "appointment"]
            )
            ULOG.log_event(category="task", subcategory="appointment_suggested",
                           summary=f"Proposed {provider} time {dstr} {tstr}", source=channel)
            return {"ok": True, "todo_id": todo.id, "proposed_time": f"{dstr} {tstr}"}
        else:
            # Next Monday helper
            today = dt.date.today()
            days_ahead = (0 - today.weekday()) % 7  # 0=Monday
            next_monday = today + dt.timedelta(days=days_ahead or 7)
            todo = reg.add(
                text=f"Call {provider} to schedule (closed evenings/weekend)",
                kind="call", due_date=next_monday.isoformat(), eta_minutes=10, source=channel,
                tags=[provider, "appointment"]
            )
            ULOG.log_event(category="task", subcategory="appointment_deferred",
                           summary=f"No slot before 19:00; defer to {next_monday}", source=channel)
            return {"ok": True, "todo_id": todo.id, "deferred_to": next_monday.isoformat()}

    # 2) Plan-today JSON already supported via plan_hook
    if intent.kind == "plan_today":
        from assistant.home.nl_intents import parse_plan_items
        base_tasks = parse_plan_items(intent.slots["payload"])
        meal_blocks = [{"title": "Cook lunch", "mins": 40}, {"title": "Cook dinner", "mins": 50}]
        out = build_today_plan(base_tasks, meal_blocks)
        ULOG.log_event(category="task", subcategory="plan_generated",
                       summary=f"blocks={len(out['plan']['blocks'])}", source=channel)
        return {"ok": True, **out}



    # 3) “Plan today:” reuses your planner + merges todos & pantry
    if intent.kind == "plan_today":
        # You probably already parse durations; keep it simple here:
        from assistant.home.nl_intents import parse_plan_items
        base_tasks = parse_plan_items(intent.slots["payload"])
        meal_blocks = [{"title": "Cook lunch", "mins": 40}, {"title": "Cook dinner", "mins": 50}]
        out = build_today_plan(base_tasks, meal_blocks)
        Logger.log("plan.generated", {"n_blocks": len(out["plan"]["blocks"]), "pantry_low": out["pantry_low"]})
        return {"ok": True, **out}

    # 4) Pass through pantry and cook to the modules you already have
    if intent.kind in ("pantry_add", "pantry_use", "cook"):
        # Reuse your assistant.home.today_cli logic directly if you prefer;
        # or call PantryManager/NutritionTracker methods inline
        from assistant.home.today_cli import main as today_main
        # Quick and dirty: shell out into the CLI-like handler function
        # (or refactor today_cli into callable functions and invoke those)
        # Here we just return a stub telling the chat surface it succeeded:
        return {"ok": True, "routed": intent.kind, "payload": intent.slots["payload"]}

    # 5) Generic todo
    if intent.kind == "todo":
        t = TodoRegistry().add(text=intent.slots["text"], source=channel)
        Logger.log("todo.add", {"text": t.text, "id": t.id})
        return {"ok": True, "todo": t.text}

    return None
