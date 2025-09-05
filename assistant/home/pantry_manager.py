from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Tuple
import json, os, math

PANTRY_PATH = os.environ.get("PANTRY_DB", "data/pantry.json")

@dataclass
class PantryItem:
    name: str
    qty: float            # current quantity
    unit: str             # e.g., "g", "ml", "pcs"
    min_qty: float = 0.0  # restock threshold in same unit
    per_unit_grams: float = 0.0  # if unit is "pcs", how many grams is one piece (for nutrition)
    # optional nutrition per 100g; leave empty to let NutritionTracker fill in
    kcal_100g: Optional[float] = None
    protein_100g: Optional[float] = None
    fat_100g: Optional[float] = None
    carbs_100g: Optional[float] = None
    micros_per_100g: Optional[Dict[str, float]] = None

class PantryManager:
    def __init__(self, path: str = PANTRY_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.items: Dict[str, PantryItem] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            data = json.load(open(self.path, "r", encoding="utf-8"))
            self.items = {k: PantryItem(**v) for k, v in data.items()}
        else:
            self._save()

    def _save(self):
        json.dump({k: asdict(v) for k, v in self.items.items()},
                  open(self.path, "w", encoding="utf-8"), indent=2)

    def add(self, name: str, qty: float, unit: str, **kwargs):
        key = name.lower().strip()
        if key in self.items and self.items[key].unit == unit:
            self.items[key].qty += qty
            # allow updating thresholds or nutrition if provided
            for k, v in kwargs.items():
                setattr(self.items[key], k, v)
        else:
            self.items[key] = PantryItem(name=key, qty=qty, unit=unit, **kwargs)
        self._save()

    def set_threshold(self, name: str, min_qty: float):
        key = name.lower().strip()
        if key not in self.items:
            raise ValueError(f"{name} not in pantry")
        self.items[key].min_qty = min_qty
        self._save()

    def use(self, name: str, qty: float):
        key = name.lower().strip()
        if key not in self.items:
            raise ValueError(f"{name} not in pantry")
        self.items[key].qty = max(0.0, self.items[key].qty - qty)
        self._save()

    def consume_bulk(self, ingredients: List[Tuple[str, float, str]]) -> List[str]:
        """Subtract ingredients. Returns list of items we’re short on (names)."""
        shortage = []
        # unit match assumed; keep units consistent in recipes
        for nm, q, unit in ingredients:
            k = nm.lower().strip()
            if k not in self.items or self.items[k].unit != unit:
                shortage.append(nm)
            else:
                if self.items[k].qty < q:
                    shortage.append(nm)
                self.items[k].qty = max(0.0, self.items[k].qty - q)
        self._save()
        return shortage

    def low_items(self) -> List[PantryItem]:
        return [item for item in self.items.values() if item.min_qty > 0 and item.qty <= item.min_qty]

    def snapshot(self) -> Dict[str, Dict]:
        return {k: asdict(v) for k, v in self.items.items()}
