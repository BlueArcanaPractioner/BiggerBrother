from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Literal
import json, os, uuid, datetime as dt

TODO_DB = os.environ.get("TODO_DB", "data/todos.json")
os.makedirs(os.path.dirname(TODO_DB), exist_ok=True)

@dataclass
class Todo:
    id: str
    text: str
    kind: Literal["call","visit","buy","task"] = "task"
    due_date: Optional[str] = None     # YYYY-MM-DD
    eta_minutes: int = 15
    source: Literal["chat","checkin","email","system"] = "system"
    tags: List[str] = None

class TodoRegistry:
    def __init__(self, path: str = TODO_DB):
        self.path = path
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            raw = json.load(open(self.path, "r"))
            self.todos = [Todo(**t) for t in raw]
        else:
            self.todos = []
            self._save()

    def _save(self):
        json.dump([asdict(t) for t in self.todos], open(self.path, "w"), indent=2)

    def add(self, text: str, kind="task", due_date: Optional[str]=None, eta_minutes=15, source="system", tags=None) -> Todo:
        t = Todo(id=str(uuid.uuid4()), text=text, kind=kind, due_date=due_date, eta_minutes=eta_minutes, source=source, tags=tags or [])
        self.todos.append(t); self._save()
        return t

    def pending_for(self, date_iso: str) -> List[Todo]:
        return [t for t in self.todos if t.due_date is None or t.due_date <= date_iso]

    def complete(self, todo_id: str):
        self.todos = [t for t in self.todos if t.id != todo_id]; self._save()
