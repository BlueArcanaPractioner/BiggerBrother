from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import json, os, datetime as dt

NUTRITION_LOG_DIR = os.environ.get("NUTRITION_LOG_DIR", "data/nutrition")

# Minimal built-in table; extend ad hoc. Values per 100g.
DEFAULT_DB = {
    "egg":         {"kcal": 143, "protein": 13,  "fat": 10,  "carbs": 1.1, "micros": {"choline_mg": 294}},
    "chicken":     {"kcal": 165, "protein": 31,  "fat": 3.6, "carbs": 0},
    "rice":        {"kcal": 130, "protein": 2.7, "fat": 0.3, "carbs": 28},
    "olive oil":   {"kcal": 884, "protein": 0,   "fat": 100, "carbs": 0, "micros": {"vitE_mg": 14.4}},
    "milk":        {"kcal": 61,  "protein": 3.2, "fat": 3.3, "carbs": 4.8, "micros": {"calcium_mg": 113}},
    "cheddar":     {"kcal": 402, "protein": 25,  "fat": 33,  "carbs": 1.3, "micros": {"calcium_mg": 721}},
    "butter":      {"kcal": 717, "protein": 0.9, "fat": 81,  "carbs": 0.1},
    "banana":      {"kcal": 89,  "protein": 1.1, "fat": 0.3, "carbs": 23, "micros": {"potassium_mg": 358}},
    "broccoli":    {"kcal": 55,  "protein": 3.7, "fat": 0.6, "carbs": 11, "micros": {"vitC_mg": 89.2}},
    "pasta":       {"kcal": 131, "protein": 5,   "fat": 1.1, "carbs": 25},
    "tomato":      {"kcal": 18,  "protein": 0.9, "fat": 0.2, "carbs": 3.9},
    "onion":       {"kcal": 40,  "protein": 1.1, "fat": 0.1, "carbs": 9.3},
    "garlic":      {"kcal": 149, "protein": 6.4, "fat": 0.5, "carbs": 33},
}

@dataclass
class NutritionTotals:
    kcal: float = 0.0
    protein: float = 0.0
    fat: float = 0.0
    carbs: float = 0.0
    micros: Dict[str, float] = None

    def add(self, other: "NutritionTotals"):
        self.kcal += other.kcal; self.protein += other.protein
        self.fat += other.fat; self.carbs += other.carbs
        if other.micros:
            if self.micros is None: self.micros = {}
            for k, v in other.micros.items():
                self.micros[k] = self.micros.get(k, 0.0) + v

class NutritionTracker:
    def __init__(self, db: Optional[Dict[str, Dict]] = None, log_dir: str = NUTRITION_LOG_DIR):
        self.db = {k.lower(): v for k, v in (db or DEFAULT_DB).items()}
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def _path_for(self, date: Optional[str] = None) -> str:
        d = date or dt.date.today().isoformat()
        return os.path.join(self.log_dir, f"{d}.json")

    def _load_day(self, date: Optional[str] = None) -> Dict:
        p = self._path_for(date)
        return json.load(open(p, "r")) if os.path.exists(p) else {"entries": [], "totals": {}}

    def _save_day(self, data: Dict, date: Optional[str] = None):
        json.dump(data, open(self._path_for(date), "w"), indent=2)

    def ensure_food(self, name: str, per100g: Dict):
        self.db[name.lower()] = per100g

    def compute(self, name: str, qty: float, unit: str, per_unit_grams: float = 0.0) -> NutritionTotals:
        # normalize quantity to grams
        nm = name.lower()
        grams = qty
        if unit in ("g", "gram", "grams"):
            grams = qty
        elif unit in ("kg",):
            grams = qty * 1000
        elif unit in ("ml",):
            # crude assumption: 1ml ≈ 1g unless otherwise specified
            grams = qty
        elif unit in ("pcs", "piece", "pieces"):
            grams = qty * (per_unit_grams or 0.0)
        else:
            raise ValueError(f"Unknown unit: {unit}")

        row = self.db.get(nm)
        if not row:
            # unknown food: log zero macros so you can backfill later
            return NutritionTotals(0,0,0,0,{})

        factor = grams / 100.0
        micros_scaled = {k: v * factor for k, v in (row.get("micros") or {}).items()}
        return NutritionTotals(
            kcal=row["kcal"] * factor,
            protein=row["protein"] * factor,
            fat=row["fat"] * factor,
            carbs=row["carbs"] * factor,
            micros=micros_scaled,
        )

    def log_meal(self, label: str, ingredients: List[Tuple[str, float, str, float]]) -> Dict:
        """
        ingredients: list of (name, qty, unit, per_unit_grams_if_pcs)
        """
        totals = NutritionTotals(0,0,0,0, {})
        parts = []
        for nm, q, unit, pug in ingredients:
            nt = self.compute(nm, q, unit, pug)
            totals.add(nt)
            parts.append({"name": nm, "qty": q, "unit": unit})

        day = self._load_day()
        day["entries"].append({"label": label, "ingredients": parts, "totals": asdict(totals)})
        agg = self.aggregate(day["entries"])
        day["totals"] = agg
        self._save_day(day)
        return day

    def aggregate(self, entries: List[Dict]) -> Dict:
        T = NutritionTotals(0,0,0,0, {})
        for e in entries:
            t = e["totals"]
            T.kcal += t["kcal"]; T.protein += t["protein"]; T.fat += t["fat"]; T.carbs += t["carbs"]
            for k, v in (t.get("micros") or {}).items():
                T.micros[k] = T.micros.get(k, 0.0) + v
        return {"kcal": round(T.kcal,1), "protein": round(T.protein,1),
                "fat": round(T.fat,1), "carbs": round(T.carbs,1), "micros": T.micros}
