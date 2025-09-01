from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json , time
import logging
import os
import re

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PROMPT_HEAD_DEFAULT = """
You are a labeling engine. Given one message (with optional metadata), produce labels as a JSON object ONLY.
Follow these rules exactly:
- Output a single JSON object with keys: topic, tone, intent, confidence.
- Each of topic/tone/intent is a list of {"label": <string>, "p": <float in [0,1]>}.
- Make labels UNIQUE by 'label' and SORTED by 'p' descending (ties: by label ascending).
- If there is no taggable text (empty or whitespace), ALL lists must be empty and confidence must be 0.0.
- Otherwise set confidence in [0,1] reflecting how certain the labels are.
- Do NOT include any extra keys or prose. JSON only.
"""

MODEL_CHAT_ONLY = ("gpt-5-nano",)  # extend if you add siblings
DEFAULT_CHAT_KWARGS = {"response_format"}  # safe everywhere; omit temp/top_p/max_tokens for nano
DEFAULT_FULL_KWARGS = {"response_format", "temperature", "top_p", "max_tokens"}

def _is_chat_model(m: str) -> bool:
    m = (m or "").lower()
    return m.startswith("gpt-") or "-turbo" in m or "-o" in m

def _allowlist_for(model: str) -> set[str]:
    m = (model or "").lower()
    if any(m.startswith(x) for x in MODEL_CHAT_ONLY):
        return DEFAULT_CHAT_KWARGS
    return DEFAULT_FULL_KWARGS

def _load_prompt_head(data_root: Path) -> str:
    # prefer a local prompt head if present
    candidates = [
        data_root / "prompt_head.txt",
        Path.cwd() / "prompt_head.txt",
    ]
    for p in candidates:
        try:
            if p.exists():
                return p.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return PROMPT_HEAD_DEFAULT.strip()

def _build_prompt(head: str, chunk: Dict[str, Any]) -> str:
    """
    Build a single-string prompt that works for both chat/completions adapters.
    """
    gid = chunk.get("gid") or ""
    role = chunk.get("role") or ""
    text = (chunk.get("content_text") or "").strip()
    conv = chunk.get("conversation_id") or chunk.get("cid") or ""
    # few-shot style example kept tiny to avoid token bloat
    example = {
        "topic": [{"label": "example", "p": 0.9}],
        "tone": [{"label": "neutral", "p": 1.0}],
        "intent": [{"label": "inform", "p": 1.0}],
        "confidence": 0.8
    }
    return (
        head
        + "\n\n"
        + "Schema:\n"
        + json.dumps({"topic":[{"label":"str","p":"float"}],"tone":[{"label":"str","p":"float"}],"intent":[{"label":"str","p":"float"}],"confidence":"float"}, indent=2)
        + "\n\n"
        + "Example:\n"
        + json.dumps(example, ensure_ascii=False, indent=2)
        + "\n\n"
        + "Now label this message. Remember: JSON only, no prose.\n"
        + json.dumps({"gid": gid, "role": role, "conversation_id": conv, "content_text": text}, ensure_ascii=False)
    )

def _extract_first_json_object(s: str):
    """
    Return the first valid top-level JSON object found in s, or None.
    Scans with a simple brace/quote state machine (handles escapes).
    """
    if not s:
        return None
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"' and depth >= 0:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    frag = s[start:i+1]
                    try:
                        return json.loads(frag)
                    except Exception:
                        # keep scanning in case there is a later valid object
                        start = -1
                        continue
    return None

def _coerce_json(s: str) -> dict:
    """
    Try json.loads; if that fails, extract the first balanced JSON object.
    Fall back to an empty, schema-shaped record.
    """
    s = (s or "").strip()
    # 1) plain parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 2) extract first balanced object from noisy text
    obj = _extract_first_json_object(s)
    if isinstance(obj, dict):
        return obj
    # 3) last resort
    return {"topic": [], "tone": [], "intent": [], "confidence": 0.0}

def _norm_list(arr: Any) -> List[Dict[str, Any]]:
    if not isinstance(arr, list):
        return []
    best: Dict[str, float] = {}
    for it in arr:
        if not isinstance(it, dict): continue
        lab = str(it.get("label", "")).strip()
        if not lab: continue
        try:
            p = float(it.get("p", 0.0))
        except Exception:
            p = 0.0
        p = 0.0 if p < 0 else (1.0 if p > 1 else p)
        if lab not in best or p > best[lab]:
            best[lab] = p
    return [{"label": k, "p": v} for k, v in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))]

def _build_prompt(head: str, chunk: dict) -> str:
    text = (chunk.get("content_text") or "").strip()
    gid  = chunk.get("gid") or ""
    example = {
        "topic": [{"label": "example", "p": 0.9}],
        "tone": [{"label": "neutral", "p": 1.0}],
        "intent": [{"label": "inform", "p": 1.0}],
        "confidence": 0.8
    }
    schema = {"topic":[{"label":"str","p":"float"}],
              "tone":[{"label":"str","p":"float"}],
              "intent":[{"label":"str","p":"float"}],
              "confidence":"float"}
    return (
        (head or "You are a labeling engine. Output JSON ONLY.")
        + "\n\nSchema:\n" + json.dumps(schema, indent=2)
        + "\n\nExample:\n" + json.dumps(example, ensure_ascii=False, indent=2)
        + "\n\nNow label this message. JSON only, no prose.\n"
        + json.dumps({"gid": gid, "content_text": text}, ensure_ascii=False)
    )


@dataclass
class LabelScore:
    label: str
    p: float


@dataclass
class LabelRecord:
    """
    Represents the label_record.schema.json structure (minimal).
    """
    gid: str
    topic: List[LabelScore] = field(default_factory=list)
    tone: List[LabelScore] = field(default_factory=list)
    intent: List[LabelScore] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable dict matching expected schema.
        Ensures uniqueness by label (keep highest p) and sorted by descending p.
        Clips probabilities and confidence to [0,1].
        """
        def normalize(lst: List[LabelScore]) -> List[Dict[str, Any]]:
            mp: Dict[str, float] = {}
            for it in lst:
                if not isinstance(it.label, str) or not it.label:
                    continue
                try:
                    p = float(it.p)
                except Exception:
                    p = 0.0
                if p < 0.0:
                    p = 0.0
                elif p > 1.0:
                    p = 1.0
                prev = mp.get(it.label)
                if prev is None or p > prev:
                    mp[it.label] = p
            sorted_items = sorted(mp.items(), key=lambda kv: kv[1], reverse=True)
            return [{"label": label, "p": p} for label, p in sorted_items]

        topic = normalize(self.topic)
        tone = normalize(self.tone)
        intent = normalize(self.intent)

        try:
            conf = float(self.confidence)
        except Exception:
            conf = 0.0
        if conf < 0.0:
            conf = 0.0
        elif conf > 1.0:
            conf = 1.0

        return {
            "gid": self.gid,
            "topic": topic,
            "tone": tone,
            "intent": intent,
            "confidence": conf
        }


def _collect_chunks(data_root: Path, manifest_glob: List[str]) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    manifests: List[Path] = []
    for pat in manifest_glob:
        p = Path(pat)
        if p.is_absolute():
            manifests.extend(q for q in p.parent.glob(p.name) if q.is_file())
        else:
            manifests.extend(q for q in data_root.glob(pat) if q.is_file())
    for mpath in sorted(set(manifests)):
        try:
            obj = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        base = mpath.parent
        for entry in obj.get("chunks", []):
            rel = entry.get("path") if isinstance(entry, dict) else entry
            if not rel:
                continue
            cpath = Path(rel)
            if not cpath.is_absolute():
                cpath = (base / rel).resolve()
            try:
                chunk = json.loads(cpath.read_text(encoding="utf-8"))
                gid = chunk.get("gid") or cpath.stem
            except Exception:
                gid = cpath.stem
            items.append((str(gid), cpath))
    # stable order
    items.sort(key=lambda t: t[0])
    return items

def main(inputs: Dict[str, Any], outputs: Dict[str, Any], openai_client: Any = None, **kwargs) -> Dict[str, Any]:
    # ---- client fallback ----
    if openai_client is None:
        try:
            from . import openai_client as oc
        except Exception:
            import openai_client as oc
        openai_client = getattr(oc, "default_client", oc)

    # ---- resolve I/O ----
    data_root = Path(inputs.get("data_root", "")).resolve()
    manifest_glob = inputs.get("manifest_glob", [])
    if isinstance(manifest_glob, (str, Path)): manifest_glob = [str(manifest_glob)]
    labels_dir = Path(outputs.get("labels_dir", "labels")).resolve()
    labels_dir.mkdir(parents=True, exist_ok=True)
    records_schema = outputs.get("records_schema", "schemas/label_record.schema.json")

    # ---- batching knobs ----
    model_hint   = inputs.get("model") or os.getenv("OPENAI_MODEL") or "gpt-5-nano"
    skip_existing = bool(inputs.get("skip_existing", True))
    limit        = int(inputs.get("limit", 0) or 0)          # max items to process
    offset       = int(inputs.get("offset", 0) or 0)         # start index after filtering
    num_shards   = int(inputs.get("num_shards", 0) or 0)     # optional shard split
    shard_index  = int(inputs.get("shard_index", 0) or 0)    # 0 <= idx < num_shards
    sleep_ms     = float(inputs.get("sleep_ms", 0) or 0.0)   # inter-call delay

    head = _load_prompt_head(data_root)

    # ---- collect all chunks ----
    all_items = _collect_chunks(data_root, manifest_glob)
    if not all_items:
        print(f"[label_generator] no manifests found under {data_root} for patterns {manifest_glob}")
        return {"labels_dir": str(labels_dir), "records_schema": records_schema, "written": [], "count": 0, "total": 0}

    # ---- filter by skip-existing ----
    def out_path_for(gid: str) -> Path: return labels_dir / f"{gid}.json"
    items = [(gid, path) for (gid, path) in all_items if not (skip_existing and out_path_for(gid).exists())]

    # ---- shard or slice ----
    if num_shards and num_shards > 1:
        # take every k-th item (round-robin) for parallel workers
        items = [item for idx, item in enumerate(items) if idx % num_shards == shard_index]
    else:
        # contiguous slice via offset/limit
        if offset > 0:
            items = items[offset:]
        if limit:
            items = items[:limit]

    processed: List[str] = []
    count = 0
    total_remaining = len(items)

    # ---- main loop (unchanged logic below this point) ----
    for gid, cpath in items:
        try:
            chunk = json.loads(cpath.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[label_generator] unreadable chunk {cpath}: {e}")
            continue

        text = (chunk.get("content_text") or "").strip()
        if not text:
            record = {"gid": gid, "topic": [], "tone": [], "intent": [], "confidence": 0.0}
        else:
            prompt = _build_prompt(head, chunk)
            try:
                if hasattr(openai_client, "chat"):
                    raw = openai_client.chat(
                        [{"role": "user", "content": prompt}],
                        model=model_hint,
                        response_format={"type": "json_object"},
                    )
                else:
                    raw = openai_client.complete(prompt=prompt, model=model_hint)
            except Exception as e:
                print(f"[label_generator] LLM call failed for {gid}: {e}")
                raw = ""

            payload = _coerce_json(raw)
            topic  = _norm_list(payload.get("topic", []))
            tone   = _norm_list(payload.get("tone", []))
            intent = _norm_list(payload.get("intent", []))
            try:
                conf = float(payload.get("confidence", 0.0) or 0.0)
            except Exception:
                conf = 0.0
            conf = 0.0 if conf < 0 else (1.0 if conf > 1.0 else conf)
            record = {"gid": gid, "topic": topic, "tone": tone, "intent": intent, "confidence": conf}

        out_path = labels_dir / f"{gid}.json"
        try:
            out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            processed.append(str(out_path))
        except Exception as e:
            print(f"[label_generator] failed writing {out_path}: {e}")

        count += 1
        if sleep_ms: time.sleep(sleep_ms / 1000.0)

    return {
        "labels_dir": str(labels_dir),
        "records_schema": str(records_schema),
        "written": processed,
        "count": count,
        "total": len(all_items),
        "skipped_existing": len(all_items) - len(items)  # rough: counts files skipped by existence test
    }