"""coverage_checker.py

Public API:
- coverage_checker.main(input: Dict[str, Any]) -> Dict[str, Any]

Responsibilities:
- discover chunk manifest files under a data_root using provided glob patterns
- load chunk(s) referenced by manifests
- for each chunk (1:1 grouping), call label_generator.main(...) to obtain label records
- normalize and validate label records to enforce schema invariants:
    - required top-level keys present (at least 'gid')
    - topic/tone/intent arrays are unique by label and sorted by descending probability
    - confidence in [0,1]
- write label records into labels_dir as JSON files (one file per gid)
- compute a coverage metric and return summary info
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import glob
import logging
import hashlib
import os

# External dependencies provided by the system (must be called exactly as specified)
try:
    from . import openai_client as oc
    from . import label_generator as lg
except ImportError:
    import openai_client as oc  # type: ignore
    import label_generator as lg  # type: ignore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
REQUIRED_FIELDS = ("gid", "topic", "tone", "intent", "confidence")

@dataclass
class ChunkRecord:
    gid: str
    role: Optional[str]
    content_text: str
    ts: Optional[str] = None
    model: Optional[str] = None
    raw_meta: Optional[Dict[str, Any]] = None


def find_manifest_files(data_root: str, manifest_glob: List[str]) -> List[Path]:
    """
    Find manifest files under data_root that match any pattern in manifest_glob.

    Returns deduplicated sorted list of Paths.
    """
    root = Path(data_root)
    seen = set()
    results: List[Path] = []
    for pattern in manifest_glob:
        # Allow patterns like "**/*.manifest.json" if provided
        search_pattern = str(root / pattern)
        for p in glob.glob(search_pattern, recursive=True):
            path = Path(p)
            if path.exists() and path.is_file():
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    results.append(resolved)
    results.sort()
    logger.debug("Found manifest files: %s", results)
    return results


def load_manifest(path: Path) -> Dict[str, Any]:
    """
    Load and parse a manifest JSON file; expects at least a "chunks" key with a list of chunk entries.
    """
    with path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict):
        raise ValueError(f"Manifest {path} did not contain a JSON object at top level.")
    if "chunks" not in doc:
        raise ValueError(f"Manifest {path} missing required 'chunks' key.")
    if not isinstance(doc["chunks"], list):
        raise ValueError(f"Manifest {path} key 'chunks' must be a list.")
    return doc


def _resolve_path(ref: str, manifest_parent: Optional[Path], data_root: Optional[Path]) -> Path:
    """
    Resolve a reference string to an existing file path. Attempts:
      - absolute path
      - relative to manifest_parent
      - relative to data_root
      - relative to cwd
    Returns resolved Path (may not exist if not found).
    """
    p = Path(ref)
    if p.is_absolute():
        return p
    if manifest_parent is not None:
        candidate = (manifest_parent / ref).resolve()
        if candidate.exists():
            return candidate
    if data_root is not None:
        candidate = (data_root / ref).resolve()
        if candidate.exists():
            return candidate
    # fallback: resolve relative to cwd
    return p.resolve()


def _parse_chunk_json(doc: Any, source_path: Optional[Path] = None) -> ChunkRecord:
    """
    Given JSON loaded from a chunk file or manifest entry (as dict), attempt to produce a ChunkRecord.
    Accepts either a dict with expected keys or a list containing a dict (first item used).
    """
    if isinstance(doc, list):
        if len(doc) == 0:
            raise ValueError(f"Chunk file {source_path} contained an empty list.")
        doc = doc[0]

    if not isinstance(doc, dict):
        raise ValueError(f"Chunk document {source_path} is not a JSON object.")

    # common key alternatives
    gid = doc.get("gid") or doc.get("id") or doc.get("message_id") or doc.get("uid")
    if not gid:
        # generate a stable gid if none present based on content
        raw = json.dumps(doc, sort_keys=True, ensure_ascii=False).encode("utf-8")
        gid = hashlib.sha1(raw).hexdigest()

    content_text = (
        doc.get("content_text")
        or doc.get("content")
        or doc.get("text")
        or doc.get("message")
        or ""
    )
    # role alternatives
    role = doc.get("role") or doc.get("speaker") or doc.get("from")
    ts = doc.get("ts") or doc.get("timestamp") or doc.get("time")
    model = doc.get("model")
    return ChunkRecord(gid=str(gid), role=role, content_text=str(content_text), ts=ts, model=model, raw_meta=doc)


def _load_chunk_entry(entry: Any, manifest_parent: Optional[Path], data_root: Optional[Path]) -> ChunkRecord:
    """
    Given an entry from a manifest's 'chunks' list, return a ChunkRecord.
    Entry may be:
      - a dict describing the chunk inline
      - a string path to a chunk file
      - a dict with a path-like key such as 'path', 'file', 'ref'
    """
    # If entry is a string, treat as path
    if isinstance(entry, str):
        p = _resolve_path(entry, manifest_parent, data_root)
        if p.exists():
            with p.open("r", encoding="utf-8") as fh:
                doc = json.load(fh)
            return _parse_chunk_json(doc, source_path=p)
        else:
            raise FileNotFoundError(f"Chunk file not found: {p}")

    if isinstance(entry, dict):
        # If dict only contains a 'path' or similar field, resolve file
        for key in ("path", "file", "ref", "chunk_path", "chunk_ref"):
            if key in entry and isinstance(entry[key], str):
                p = _resolve_path(entry[key], manifest_parent, data_root)
                if p.exists():
                    with p.open("r", encoding="utf-8") as fh:
                        doc = json.load(fh)
                    return _parse_chunk_json(doc, source_path=p)
                else:
                    raise FileNotFoundError(f"Chunk file not found: {p}")
        # Otherwise treat the dict as the chunk content itself
        return _parse_chunk_json(entry, source_path=manifest_parent)
    # unknown entry type
    raise ValueError("Unsupported chunk entry type in manifest.")


def _normalize_label_array(arr: Any) -> List[Dict[str, Any]]:
    """
    Normalize a label array so that items are dicts with keys 'label' and 'probability' (float).
    Ensures uniqueness by label (keeps highest probability) and sorts descending by probability.
    Returns an array of dicts: [{'label': str, 'probability': float}, ...]
    """
    if not isinstance(arr, list):
        return []

    tmp: Dict[str, float] = {}
    for it in arr:
        if isinstance(it, dict):
            label = it.get("label") or it.get("name") or it.get("tag")
            prob = it.get("probability")
            if prob is None:
                # try alternate keys
                prob = it.get("score") or it.get("confidence")
            try:
                prob_val = float(prob) if prob is not None else 0.0
            except Exception:
                prob_val = 0.0
            if label is None:
                # skip unlabeled entries
                continue
            label_str = str(label)
            # keep max probability for duplicates
            prev = tmp.get(label_str)
            if prev is None or prob_val > prev:
                tmp[label_str] = prob_val
        else:
            # If item is a string, treat as label with default probability 1.0
            if isinstance(it, str):
                if it not in tmp:
                    tmp[it] = 1.0
    # build sorted list
    items = [{"label": k, "probability": float(v)} for k, v in tmp.items()]
    items.sort(key=lambda x: x["probability"], reverse=True)
    return items


def _normalize_record(raw: Dict[str, Any], chunk: ChunkRecord) -> Dict[str, Any]:
    """
    Normalize the raw record returned by label_generator:
      - ensure 'gid' exists and matches chunk gid when possible
      - normalize arrays under keys 'topic', 'tone', 'intent' if present
      - compute/ensure 'confidence' between 0 and 1
    """
    rec = dict(raw)  # shallow copy
    # Ensure gid
    if "gid" not in rec or not rec.get("gid"):
        rec["gid"] = chunk.gid
    else:
        # coerce to string
        rec["gid"] = str(rec["gid"])

    # Normalize known label arrays
    for key in ("topic", "tone", "intent"):
        if key in rec:
            rec[key] = _normalize_label_array(rec.get(key))
    # Some generators may put labels under 'labels' keyed by type
    if "labels" in rec and isinstance(rec["labels"], dict):
        labels_obj = rec["labels"]
        for key in ("topic", "tone", "intent"):
            if key in labels_obj:
                rec[key] = _normalize_label_array(labels_obj.get(key))
        # If 'labels' contains a flat list of dicts with 'type' and 'label', convert to arrays
        if isinstance(labels_obj, dict):
            # attempt to extract typed labels if present
            typed = {}
            for k, v in labels_obj.items():
                if k not in ("topic", "tone", "intent"):
                    # might be entries with type in item
                    if isinstance(v, list):
                        for it in v:
                            if isinstance(it, dict):
                                t = it.get("type")
                                l = it.get("label")
                                p = it.get("probability") or it.get("score") or it.get("confidence")
                                if t and l:
                                    typed.setdefault(t, []).append({"label": l, "probability": p})
            for t, arr in typed.items():
                rec[t] = _normalize_label_array(arr)

    # Compute confidence: if provided, clamp to [0,1]; otherwise derive from max probability of arrays
    conf = rec.get("confidence")
    conf_val: Optional[float] = None
    try:
        if conf is not None:
            conf_val = float(conf)
    except Exception:
        conf_val = None
    if conf_val is None:
        maxp = 0.0
        for key in ("topic", "tone", "intent"):
            arr = rec.get(key)
            if isinstance(arr, list) and len(arr) > 0:
                try:
                    p0 = float(arr[0].get("probability", 0.0))
                except Exception:
                    p0 = 0.0
                if p0 > maxp:
                    maxp = p0
        conf_val = maxp
    # clamp
    if conf_val is None:
        conf_val = 0.0
    if conf_val < 0.0:
        conf_val = 0.0
    if conf_val > 1.0:
        conf_val = 1.0
    rec["confidence"] = conf_val

    return rec


def _safe_gid_filename(gid: str) -> str:
    """
    Produce a filesystem-safe filename for a gid. If gid looks safe (alnum,-,_,.), use as-is;
    otherwise hash it.
    """
    # allow a-zA-Z0-9-_. characters
    safe = "".join(ch for ch in gid if ch.isalnum() or ch in "-_.")
    if not safe or len(safe) > 200:
        # fallback to sha1
        return hashlib.sha1(gid.encode("utf-8")).hexdigest()
    return safe

def main(inputs: dict, outputs: dict | None = None, openai_client=None, **kwargs):
    if openai_client is None:
        # safe fallback if caller didn’t pass a client
        openai_client = getattr(oc, "default_client", oc)
    """
    Validate label coverage for the given manifests.
    Optionally autogenerate missing labels by calling label_generator ONCE.
    """
    # ---- resolve I/O (absolute where possible) ----
    data_root = Path(inputs.get("data_root", "")).resolve()
    manifest_glob = inputs.get("manifest_glob", ["chunk_manifest.json", "*.manifest.json", "*.manifest"])
    if isinstance(manifest_glob, (str, Path)):
        manifest_glob = [str(manifest_glob)]

    outputs = outputs or {}
    labels_dir = Path(outputs.get("labels_dir", os.environ.get("LABELS_DIR", "labels"))).resolve()
    records_schema = outputs.get("records_schema", "schemas/label_record.schema.json")
    labels_dir.mkdir(parents=True, exist_ok=True)

    # ---- find chunk list ----
    manifests = _find_manifest_files(data_root, manifest_glob)
    chunks = _collect_chunks(manifests, data_root)
    total = len(chunks)

    # ---- check coverage (no writes here) ----
    missing: List[str] = []
    labeled = 0

    for gid, cpath in chunks:
        out_path = labels_dir / f"{_safe_gid(gid)}.json"  # <- single truth for filenames
        if not out_path.exists():
            missing.append(gid)
            continue
        try:
            rec = json.loads(out_path.read_text(encoding="utf-8"))
            if _is_labeled(rec):
                labeled += 1
            else:
                missing.append(gid)
        except Exception as e:
            logger.warning("Unreadable label file %s: %s", out_path, e)
            missing.append(gid)

    # ---- optional: autogenerate missing (call generator ONCE) ----
    if missing and kwargs.get("autogenerate_missing"):
        try:
            from . import label_generator  # or plain import if in same pkg
        except Exception:
            import label_generator  # fallback
        lg_inputs = {"data_root": str(data_root), "manifest_glob": manifest_glob}
        lg_outputs = {"labels_dir": str(labels_dir), "records_schema": records_schema}
        _ = lg.main(lg_inputs, lg_outputs, openai_client) # generator writes files

        # re-check coverage after generation
        labeled = 0
        still_missing: List[str] = []
        for gid, _ in chunks:
            out_path = labels_dir / f"{_safe_gid(gid)}.json"
            if not out_path.exists():
                still_missing.append(gid)
                continue
            try:
                rec = json.loads(out_path.read_text(encoding="utf-8"))
                if _is_labeled(rec):
                    labeled += 1
                else:
                    still_missing.append(gid)
            except Exception:
                still_missing.append(gid)
        missing = still_missing

    coverage = (labeled / total) if total else 0.0
    return {
        "labels_dir": str(labels_dir),
        "records_schema": records_schema,
        "coverage": coverage,
        "total": total,
        "labeled": labeled,
        "missing": missing,
    }

# ---- helpers ----
def _find_manifest_files(root: Path, patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(root.glob(pat)))
    return files

def _collect_chunks(manifests: List[Path], root: Path) -> List[tuple[str, Path]]:
    out: List[tuple[str, Path]] = []
    for m in manifests:
        try:
            obj = json.loads(m.read_text(encoding="utf-8"))
        except Exception:
            continue
        base = m.parent
        for entry in obj.get("chunks", []):
            p = entry.get("path")
            if not p:
                continue
            cpath = (base / p) if not os.path.isabs(p) else Path(p)
            try:
                chunk = json.loads(Path(cpath).read_text(encoding="utf-8"))
                gid = chunk.get("gid") or Path(cpath).stem
                out.append((gid, Path(cpath)))
            except Exception:
                gid = Path(cpath).stem
                out.append((gid, Path(cpath)))
    return out

def _safe_gid(gid: str) -> str:
    return "".join(ch for ch in str(gid) if ch.isalnum() or ch in ("-", "_", ".")) or "unknown"

def _is_labeled(rec: dict) -> bool:
    # required fields present
    if not all(k in rec for k in REQUIRED_FIELDS):
        return False
    # any non-empty label array
    for key in ("topic", "tone", "intent"):
        if isinstance(rec.get(key), list) and len(rec[key]) > 0:
            return True
    return False