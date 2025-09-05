from __future__ import annotations
import sys, json
from .pantry_manager import PantryManager
from .nutrition_tracker import NutritionTracker
from .day_planner import DayPlanner
from .nl_intents import ADD_RE, USE_RE, THR_RE, COOK_RE, PLAN_RE, SHOW_RE, parse_cook_payload, parse_plan_items

def main():
    pantry = PantryManager()
    nutrition = NutritionTracker()
    planner = DayPlanner()

    # Read whole command from argv or stdin
    text = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else sys.stdin.read().strip()

    # pantry add
    m = ADD_RE.match(text)
    if m:
        qty, unit, name = float(m.group(1)), m.group(2), m.group(3)
        pantry.add(name, qty, unit)
        return _ok({"pantry": pantry.snapshot()})

    # pantry use
    m = USE_RE.match(text)
    if m:
        qty, unit, name = float(m.group(1)), m.group(2), m.group(3)
        pantry.use(name, qty)
        return _ok({"pantry": pantry.snapshot()})

    # threshold
    m = THR_RE.match(text)
    if m:
        name, min_qty, unit = m.group(1), float(m.group(2)), m.group(3)
        # ensure the unit exists already for the item
        pantry.set_threshold(name, min_qty)
        return _ok({"pantry": pantry.snapshot()})

    # cook / cooking
    m = COOK_RE.match(text)
    if m:
        label, ingredients = parse_cook_payload(m.group(2))
        # subtract from pantry
        shortage = pantry.consume_bulk([(nm, q, unit) for nm, q, unit in ingredients])
        # log nutrition (assume per_unit_grams=0 unless “pcs” known on item)
        enrich = []
        for nm, q, unit in ingredients:
            it = pantry.items.get(nm.lower())
            pug = 0.0 if not it else it.per_unit_grams
            enrich.append((nm, q, unit, pug))
        day = nutrition.log_meal(label, enrich)
        low = [i.name for i in pantry.low_items()]
        return _ok({"shortage": shortage, "low": low, "today_totals": day["totals"]})

    # plan today
    m = PLAN_RE.match(text)
    if m:
        tasks = parse_plan_items(m.group(1))
        low = [i.name for i in pantry.low_items()]
        meal_blocks = [{"title":"Cook lunch","mins":40}, {"title":"Cook dinner","mins":50}]
        plan = planner.plan(tasks, low, meal_blocks)
        return _ok({"plan": plan})

    # show pantry/macros
    m = SHOW_RE.match(text)
    if m and m.group(1).lower() == "pantry":
        return _ok({"pantry": pantry.snapshot()})
    if m and m.group(1).lower() == "macros":
        import datetime as dt, os, json
        p = f"data/nutrition/{dt.date.today().isoformat()}.json"
        return _ok(json.load(open(p)) if os.path.exists(p) else {"entries": [], "totals": {}})

    return _err("Unrecognized command. Try:\n"
                "  plan today: deep work for 90m; inbox zero for 30m\n"
                "  pantry add 12 pcs egg\n"
                "  pantry set threshold milk 500 ml\n"
                "  cook: 2 egg, 30 g cheddar, 10 g butter\n"
                "  show pantry | show macros")

def _ok(payload): print(json.dumps({"status":"ok", **payload}, indent=2))
def _err(msg):   print(json.dumps({"status":"error","message":msg}, indent=2))

if __name__ == "__main__":
    main()
