from __future__ import annotations
from typing import Dict, List, Tuple
import re

# Patterns:
#  - plan today: <task1>[for Xm]; <task2>...
#  - pantry add 2 pcs egg
#  - pantry set threshold milk 500 ml
#  - cook: 2 egg, 30 g cheddar, 10 g butter
#  - cooking <label>: (alias of cook)
#  - pantry use 200 g rice
#  - show pantry / show macros

ADD_RE = re.compile(r"^pantry\s+add\s+([\d.]+)\s*(\w+)\s+(.+)$", re.I)
USE_RE = re.compile(r"^pantry\s+use\s+([\d.]+)\s*(\w+)\s+(.+)$", re.I)
THR_RE = re.compile(r"^pantry\s+set\s+threshold\s+(.+?)\s+([\d.]+)\s*(\w+)$", re.I)
COOK_RE = re.compile(r"^(cook|cooking)\s*:?(.+)$", re.I)
PLAN_RE = re.compile(r"^plan\s+today\s*:(.+)$", re.I)
SHOW_RE = re.compile(r"^show\s+(pantry|macros)$", re.I)

def parse_cook_payload(s: str) -> Tuple[str, List[Tuple[str,float,str]]]:
    """
    "omelette: 2 egg, 30 g cheddar, 10 g butter"
    or just "2 egg, 30 g cheddar" (label inferred)
    """
    if ":" in s:
        label, rest = [x.strip() for x in s.split(":", 1)]
    else:
        label, rest = "meal", s.strip()
    parts = []
    for piece in rest.split(","):
        piece = piece.strip()
        m = re.match(r"([\d.]+)\s*(\w+)\s+(.+)", piece)
        if not m: continue
        qty, unit, name = float(m.group(1)), m.group(2), m.group(3).strip()
        parts.append((name, qty, unit))
    return label, parts

def parse_plan_items(s: str) -> List[Dict]:
    """
    "deep work for 90m; inbox zero for 30m; guitar practice"
    """
    items = []
    for chunk in s.split(";"):
        t = chunk.strip()
        if not t: continue
        from .day_planner import parse_duration_mins
        mins = parse_duration_mins(t, default=50)
        title = re.sub(r"\bfor\s+\d+\s*(m|min|minutes|h|hr|hours)\b", "", t, flags=re.I).strip(" ,")
        items.append({"title": title, "mins": mins, "kind": "work"})
    return items
