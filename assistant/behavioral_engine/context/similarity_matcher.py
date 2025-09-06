"""
Context Similarity Matcher for BiggerBrother
============================================

Implements weighted similarity matching between messages using harmonized labels,
tier weights, and recency bias to build optimal context headers.
"""

from __future__ import annotations
import json
import os
import sys
import math
import time
import hashlib
import platform
import threading
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
import re
import tempfile, pickle
import tracemalloc
import faulthandler
from types import SimpleNamespace
from functools import lru_cache

from numpy import save

from app.openai_client import OpenAIClient
from app.label_integration_wrappers import LabelGenerator
from assistant.importers.enhanced_smart_label_importer import EnhancedLabelHarmonizer

LOGGER_NAME = "matcher"

def _ts():
    return datetime.now(timezone.utc).isoformat()

def _log(msg: str):
    print(f"[{LOGGER_NAME}][{_ts()}] {msg}")


# --- robust label normalization ---------------------------------------------
def _as_label_list(val: Any) -> List[str]:
    """
    Convert various shapes into a clean list[str] of *labels*.
    - Strings -> [string]
    - (list|tuple|set) of strings -> list[str]
    - NumPy arrays:
        * object/str dtype -> tolist()  str()
        * numeric dtype (e.g., embeddings) -> ignored (return [])
    - dict: if it looks like {'labels': [...]}, use that; else ignore ([])
    - None/other -> []
    """
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, (list, tuple, set)):
        return [str(x) for x in val if isinstance(x, str)]
    if isinstance(val, dict):
        lab = val.get("labels")
        if isinstance(lab, (list, tuple, set, np.ndarray)):
            return _as_label_list(lab)
        return []
    try:
        if isinstance(val, np.ndarray):
            # treat only string/object arrays as label-y; ignore numeric arrays (e.g., embeddings)
            if val.dtype.kind in ("U", "S", "O"):
                return [str(x) for x in val.tolist()]
            return []
    except Exception:
        pass
    return []

# --- perf flags (all optional, toggle via env vars) ---
_PROFILE      = os.getenv("BB_MATCHER_PROFILE", "0") == "1"
_LOG_EVERY    = int(os.getenv("BB_MATCHER_LOG_EVERY", "500"))   # loop progress cadence
_STALL_SECS   = int(os.getenv("BB_MATCHER_STALL_SECS", "120"))  # dump tracebacks if no log in this long
_MEM_SAMPLING = os.getenv("BB_MATCHER_MEM", "0") == "1"
_IO_WARN_SECS = float(os.getenv("BB_MATCHER_TO_WARN_SECS", "2.0"))
# Hard watchdog (faulthandler) is disabled on Windows to avoid access violations on 3.13
_USE_HARD_FAULT = (_PROFILE and platform.system() != "Windows")

class PhaseTimer:
    def __init__(self, name: str, sink: Dict[str, float]):
        self.name = name
        self.sink = sink
    def __enter__(self):
        self.t0 = time.perf_counter()
        if _PROFILE: _log(f"BEGIN {self.name}")
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        self.sink[self.name] = self.sink.get(self.name, 0.0) + dt
        if _PROFILE: _log(f"END   {self.name} in {dt:.3f}s")


class ContextSimilarityMatcher:
    """
    Matches messages based on harmonized label similarity with tier weighting
    and recency bias for optimal context building.
    """
    
    def __init__(
        self,
        labels_dir: str = "labels",
        chunks_dir: str = "data/chunks", 
        harmonization_dir: str = "data",
        openai_client: Optional[OpenAIClient] = None,
        harmonizer: Optional[EnhancedLabelHarmonizer] = None,
        # Context size parameters
        context_minimum_char_long_term: int = 150000,
        context_minimum_char_recent: int = 50000,
        max_context_messages: int = 50000,
        # Weighting parameters
        general_tier_weight: float = 0.3,
        specific_tier_weight: float = 0.7,
        recency_decay_factor: float = 0.998,  # Per day decay
        recency_cutoff_days: int = 1000,
    ):
        """
        Initialize the similarity matcher.
        
        Args:
            labels_dir: Directory containing label files
            chunks_dir: Directory containing chunk files
            harmonization_dir: Directory containing harmonization files
            openai_client: OpenAI client for new label generation
            context_minimum_char_long_term: Min chars for long-term context
            context_minimum_char_recent: Min chars for recent context
            max_context_messages: Maximum number of messages to include
            general_tier_weight: Weight for general tier matches
            specific_tier_weight: Weight for specific tier matches
            recency_decay_factor: Exponential decay per day for recency
            recency_cutoff_days: Max days to look back for context
        """
        self._perf = {}
        self._metrics = SimpleNamespace(sim_calls = 0, sim_time=0.0, io_calls = 0, io_time=0.0)
        self._soft_wd = None #soft watchdog handle

        #chunk index/cache
        self._chunk_index: Optional[Dict[str, Path]] = None
        self._chunk_cache: Dict[str, Dict] = {}

        self.labels_dir = Path(labels_dir)
        self.chunks_dir = Path(chunks_dir)
        self.harmonization_dir = Path(harmonization_dir)
        
        self.openai_client = openai_client or OpenAIClient()
        self.label_generator = LabelGenerator(self.openai_client)
        self.harmonizer = harmonizer

        # Embedding cache file: reuse harmonizer's if available, else local default
        if self.harmonizer and hasattr(self.harmonizer, "embedding_cache_file"):
            self.embedding_cache_file = Path(self.harmonizer.embedding_cache_file)
        else:
            self.embedding_cache_file = self.harmonization_dir / 'embedding_cache.pkl'

        # Write-through persistence behavior for harmonization updates.
        # Set BB_WRITE_THROUGH_HARMONIZATION=0 to disable immediate writes.
        self.write_through_harmonization = bool(int(os.getenv("BB_WRITE_THROUGH_HARMONIZATION", "1")))
        # tiny flag placeholder if you later want to coalesce writes
        self._harm_pending = {"general": False, "specific": False}

        
        # Context parameters
        self.context_minimum_char_long_term = context_minimum_char_long_term
        self.context_minimum_char_recent = context_minimum_char_recent
        self.max_context_messages = max_context_messages
        
        # Weighting parameters
        self.general_tier_weight = general_tier_weight
        self.specific_tier_weight = specific_tier_weight
        self.recency_decay_factor = recency_decay_factor
        self.recency_cutoff_days = recency_cutoff_days
        
        # Load harmonization groups
        self.general_groups = self._load_harmonization_tier("general")
        self.specific_groups = self._load_harmonization_tier("specific")
        
        # Build reverse mappings for fast lookup
        self.general_mapping = self._build_reverse_mapping(self.general_groups)
        self.specific_mapping = self._build_reverse_mapping(self.specific_groups)
        
        # Cache for embeddings
        self.embedding_cache = {}
        self._load_embedding_cache()

        # Debug knobs (env-driven so you don't have to change code)
        self.debug_verbose = os.getenv("BB_DEBUG_VERBOSE", "0") == "1"
        self.debug_string_only = os.getenv("BB_MATCHER_STRING_ONLY", "0") == "1"
        self.debug_max_groups = int(os.getenv("BB_DEBUG_MAX_GROUPS", "500"))
        self.debug_step = int(os.getenv("BB_DEBUG_STEP", "50"))

        self.gen_thr = float(os.getenv("BB_THR_GENERAL", "0.5"))
        self.spec_thr = float(os.getenv("BB_THR_SPECIFIC", "0.8"))
        self.gen_thr_str = float(os.getenv("BB_THR_GENERAL_STR", "0.35"))
        self.spec_thr_str = float(os.getenv("BB_THR_SPECIFIC_STR", "0.6"))

        # Embedding blend / rerank controls
        self.use_embedding_blend = os.getenv("BB_EMBED_BLEND", "1") == "1"
        self.embed_blend_weight = float(os.getenv("BB_EMBED_WEIGHT", "0.6"))
        self.embed_rerank_topk = int(os.getenv("BB_EMBED_RERANK_TOPK", "300"))
        # Never hit network during rerank unless explicitly allowed
        self.embed_rerank_allow_api = os.getenv("BB_EMBED_RERANK_API", "0") == "1"
        # Embedding model + dimension (must be consistent everywhere)
        self.embed_model = os.getenv("BB_EMBED_MODEL", os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
        _dim_map = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
        self.embed_dim = int(os.getenv("BB_EMBED_DIM", str(_dim_map.get(self.embed_model, 1536))))
        # New: never blend embeddings during the main scan unless explicitly enabled
        self.embed_in_scan = os.getenv("BB_EMBED_IN_SCAN", "0") == "1"
        # New: control how often we write the embedding cache to disk
        #   immediate | throttle | off
        self.embed_cache_save_mode = os.getenv("BB_EMBED_CACHE_SAVE", "throttle")
        self.embed_cache_save_every = int(os.getenv("BB_EMBED_CACHE_SAVE_EVERY", "200"))
        self.embed_cache_save_secs = float(os.getenv("BB_EMBED_CACHE_SAVE_SECS", "5.0"))
        self._embed_cache_dirty = 0
        self._embed_cache_last_save = time.time()

        def _dbg(msg: str):
            if self.debug_verbose:
                print(f"[matcher][{datetime.now(timezone.utc).isoformat()}] {msg}")
        self._dbg = _dbg


        # Light-weight synonym map (can be extended via JSON file)
        self.synonyms = {
            # intent (general)
            "schedule_meeting": "scheduling",
            "set_appointment": "scheduling",
            "arrange_meeting": "scheduling",
            "book_appointment": "scheduling",
            "reschedule": "scheduling",
            "calendar_event": "scheduling",
        }
        syn_file = Path(os.getenv("BB_LABEL_SYNONYMS", "data/label_synonyms.json"))
        if syn_file.exists():
            try:
                self.synonyms.update(json.loads(syn_file.read_text(encoding="utf-8")))
                self._dbg(f"loaded {len(self.synonyms)} synonyms")
            except Exception as e:
                self._dbg(f"failed to load synonyms: {e}")
        # Optional: debug sampling of canonical keys
        self.debug_list_keys = int(os.getenv("BB_DEBUG_LIST_KEYS", "0"))
        self.debug_find_contains = os.getenv("BB_DEBUG_FIND_CONTAINS", "").lower().strip()

    # ----------------- soft watchdog (safe on Windows) -----------------
    def _start_soft_watchdog(self, stall_secs: int):
        state = SimpleNamespace(last=time.perf_counter(), stop=False, secs=stall_secs)
        def loop():
            while not state.stop:
                time.sleep(stall_secs)
                nowt = time.perf_counter()
                if nowt - state.last >= stall_secs:
                    _log(f"SOFT TIMEOUT ({stall_secs}s)! Dumping stacks …")
                    for tid, frame in sys._current_frames().items():
                        print(f"\n--- Thread {tid} stack ---")
                        traceback.print_stack(frame)
                    print("--- end stacks ---\n")
        th = threading.Thread(target=loop, daemon=True)
        th.start()
        self._soft_wd = SimpleNamespace(state=state, thread=th)

    # ----------------- fast chunk lookup -----------------
    def _ensure_chunk_index(self):
        """Build an in-memory gid -> chunk_path index once."""
        if self._chunk_index is not None:
            return
        idx: Dict[str, Path] = {}
        t0 = time.perf_counter()

        # Primary dir + optional extra dirs (use ; or : to separate)
        extra = os.getenv("BB_MATCHER_EXTRA_CHUNK_DIRS", "")
        scan_dirs = [self.chunks_dir] + [Path(p) for p in re.split(r"[;:]", extra) if p.strip()]
        seen: set[str] = set()

        for root in scan_dirs:
            try:
                root = Path(root)
            except Exception:
                continue
            if not root.exists():
                continue

            for chunk_file in root.glob("*.json"):
                try:
                    with open(chunk_file, "r", encoding="utf-8-sig") as f:
                        obj = json.load(f)
                except Exception:
                    continue

                # 1) native ChunkRecord gid (our writer)
                gids: list[str] = []
                g = obj.get("gid")
                if g:
                    gids.append(str(g))

                # 2) export shape: conversation_id + seq -> "{conversation_id}#{seq:06d}"
                conv = obj.get("conversation_id")
                seq = obj.get("seq")
                if conv is not None and seq is not None:
                    try:
                        seq_i = int(seq)
                        gids.append(f"{conv}#{seq_i:06d}")
                    except Exception:
                        gids.append(f"{conv}#{seq}")

                # 3) filename-as-gid fallback for exports already named like "...#000019.json"
                stem = chunk_file.stem
                if "#" in stem:
                    gids.append(stem)

                for gid in gids:
                    if not gid or gid in seen:
                        continue
                    idx[gid] = chunk_file
                    seen.add(gid)

        self._chunk_index = idx
        if _PROFILE:
            _log(f"built chunk index: {len(idx)} files in {time.perf_counter()-t0:.2f}s")

    def _soft_watchdog_heartbeat(self):
        if self._soft_wd is not None:
            self._soft_wd.state.last = time.perf_counter()

    def _stop_soft_watchdog(self):
        if self._soft_wd is not None:
            self._soft_wd.state.stop = True
            self._soft_wd = None

    def _load_harmonization_tier(self, tier: str) -> Dict:
        """Load a harmonization tier file."""
        file_path = self.harmonization_dir / f"harmonization_{tier}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "threshold": 0.5 if tier == "general" else 0.8,
            "tier": tier,
            "description": f"{tier.capitalize()} tier harmonization",
            "topic": {"groups": {}, "group_count": 0, "total_labels": 0},
            "tone": {"groups": {}, "group_count": 0, "total_labels": 0},
            "intent": {"groups": {}, "group_count": 0, "total_labels": 0}
        }
    
    def _build_reverse_mapping(self, harmonization: Dict) -> Dict[str, Dict[str, str]]:
        """Build reverse mapping from label to canonical group."""
        mapping = {"topic": {}, "tone": {}, "intent": {}}
        
        for category in ["topic", "tone", "intent"]:
            groups = harmonization.get(category, {}).get("groups", {})
            for canonical, variants in groups.items():
                for variant in variants:
                    mapping[category][variant] = canonical
        
        return mapping
    
    def _load_embedding_cache(self):
        """Load cached embeddings if available."""
        cache_file = getattr(self, "embedding_cache_file", self.harmonization_dir / "embedding_cache.pkl")
        if cache_file.exists():
            import pickle
            try:
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
            except:
                self.embedding_cache = {}

    def _save_embedding_cache(self):
        """Save embedding cache atomically."""
        cache_file = getattr(self, "embedding_cache_file", self.harmonization_dir / "embedding_cache.pkl")
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            dir=cache_file.parent, prefix=cache_file.name, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(self.embedding_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                if os.getenv("BB_EMBED_CACHE_FSYNC", "0") == "1":
                    os.fsync(f.fileno())
            os.replace(tmp_path, cache_file)  # atomic on same filesystem
        finally:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass

    def _ensure_embeddings_for(self, labels: List[str], save_every: int = 1, allow_api: bool = True) -> None:
        """
        Ensure embeddings exist for each label in the persistent cache.
        """
        added = 0
        for i, lab in enumerate(labels, 1):
            if lab in self.embedding_cache:
                continue
            v = self._get_embedding(lab, allow_api=allow_api, save=False)
            if v is not None:
                added += 1
                if added % save_every == 0:
                    try:
                        self._save_embedding_cache()
                    except Exception:
                        pass
        if added and self.embed_cache_save_mode != "off":
            try:
                self._save_embedding_cache()
                self._embed_cache_dirty = 0
                self._embed_cache_last_save = time.time()
            except Exception:
                pass
    
    def get_message_labels(self, message: str) -> Dict:
        """
        Get labels for a message, either generating new ones or using cached.
        
        Args:
            message: The message text
            
        Returns:
            Dictionary with topic, tone, intent labels
        """
        # Generate labels using the label generator
        labels = self.label_generator.generate_labels_for_text(message)
        
        # Structure should match our label schema
        if not all(k in labels for k in ["topic", "tone", "intent"]):
            # Ensure proper structure
            labels = {
                "topic": labels.get("topic", []),
                "tone": labels.get("tone", []),
                "intent": labels.get("intent", []),
                "confidence": labels.get("confidence", 0.5)
            }
        
        return labels
    
    def harmonize_and_score_labels(self, labels: Dict) -> Dict:
        """
        Harmonize labels and calculate group scores.
        
        Args:
            labels: Raw labels with probabilities
            
        Returns:
            Dictionary with harmonized groups and scores
        """
        harmonized = {
            "general": {"topic": {}, "tone": {}, "intent": {}},
            "specific": {"topic": {}, "tone": {}, "intent": {}}
        }
        
        for category in ["topic", "tone", "intent"]:
            category_labels = labels.get(category, [])
            
            for label_item in category_labels:
                label = label_item.get("label", "")
                prob = label_item.get("p", 1.0)
                
                # Check general tier
                general_group = self.general_mapping[category].get(label)
                if not general_group:
                    # Find best matching group
                    general_group = self._find_best_group(label, category, "general")
                    if general_group:
                        self._add_to_group(label, general_group, category, "general")
                
                if general_group:
                    if general_group not in harmonized["general"][category]:
                        harmonized["general"][category][general_group] = {
                            "count": 0,
                            "total_prob": 0.0,
                            "labels": []
                        }
                    harmonized["general"][category][general_group]["count"] += 1
                    harmonized["general"][category][general_group]["total_prob"] += prob
                    harmonized["general"][category][general_group]["labels"].append(label)
                
                # Check specific tier
                specific_group = self.specific_mapping[category].get(label)
                if not specific_group:
                    specific_group = self._find_best_group(label, category, "specific")
                    if specific_group:
                        self._add_to_group(label, specific_group, category, "specific")
                
                if specific_group:
                    if specific_group not in harmonized["specific"][category]:
                        harmonized["specific"][category][specific_group] = {
                            "count": 0,
                            "total_prob": 0.0,
                            "labels": []
                        }
                    harmonized["specific"][category][specific_group]["count"] += 1
                    harmonized["specific"][category][specific_group]["total_prob"] += prob
                    harmonized["specific"][category][specific_group]["labels"].append(label)
        
        # Calculate scores for each group
        for tier in ["general", "specific"]:
            for category in ["topic", "tone", "intent"]:
                for group, data in harmonized[tier][category].items():
                    # Score = average_probability * occurrence_count
                    avg_prob = data["total_prob"] / data["count"] if data["count"] > 0 else 0
                    data["score"] = avg_prob * data["count"]
        
        return harmonized

    def _specifics_under_general(self, category: str, general_canon: str) -> set[str]:
        """
        Collect the set of specific canonical labels that live under a given general group.
        We derive it by mapping that general group's member *variants* through self._canon_map.
        """
        variants = (self.general_groups.get(category, {})
                    .get("groups", {})
                    .get(general_canon, []))
        allowed = set()
        cmap = self._canon_map.get(category, {})
        for raw in variants:
            spec, _gen = cmap.get(raw, (None, None))
            if spec:
                allowed.add(spec)
        return allowed

    def _find_best_specific_within(self, label: str, category: str, general_canon: str) -> str | None:
        """
        Use your _find_best_group but restricted to specifics that belong to the chosen general.
        We do this by temporarily filtering the group dict to the allowed specific canonicals.
        """
        allowed = self._specifics_under_general(category, general_canon)
        if not allowed:
            return None

        # clone a minimal view of specific groups limited to `allowed`
        groups_all = self.specific_groups.get(category, {}).get("groups", {})
        orig = groups_all  # keep a handle

        # build a restricted dict: only allowed canonicals
        restricted = {k: v for k, v in groups_all.items() if k in allowed}

        # temporarily swap just for this call
        self.specific_groups[category]["groups"] = restricted
        try:
            return self._find_best_group(label, category, "specific")
        finally:
            # restore full view
            self.specific_groups[category]["groups"] = orig
    
    def _find_best_group(self, label: str, category: str, tier: str) -> Optional[str]:
        """
        Find the best matching group for a label using embeddings.
        
        Args:
            label: The label to match
            category: topic, tone, or intent
            tier: general or specific
            
        Returns:
            Best matching group or None
        """
        threshold = (self.gen_thr if tier == "general" else self.spec_thr)
        if self.debug_string_only:
            threshold = (self.gen_thr_str if tier == "general" else self.spec_thr_str)
        groups = self.general_groups if tier == "general" else self.specific_groups
        category_groups = groups.get(category, {}).get("groups", {})
        
        if not category_groups:
            return None

        t0 = time.time()
        n_groups = len(category_groups)
        self._dbg(
            f"_find_best_group start tier={tier} cat={category} label='{label}' groups={n_groups} thr={threshold} string_only={self.debug_string_only}")
        # 0) synonym shortcut (string-normalized)
        norm_label = self._normalize_label(label)
        syn_target = self.synonyms.get(norm_label)
        if syn_target and syn_target in category_groups:
            self._dbg(f"_find_best_group: synonym shortcut '{label}' -> '{syn_target}'")
            return syn_target
        # nearest-canonical fallback if the synonym target doesn't exist as a key
        if syn_target:
            tgt_norm = self._normalize_label(syn_target)
            best_canon = None; best_sim = 0.0
            T = self._tokens(tgt_norm)
            if T:
                for canon in category_groups.keys():
                    sim = self._token_jaccard(T, self._tokens(self._normalize_label(canon)))
                    if sim > best_sim:
                        best_sim, best_canon = sim, canon
                # accept if it's reasonably close (half of threshold is fine for a “hint”)
                min_accept = (self.gen_thr_str if self.debug_string_only else self.gen_thr) * 0.5
                if best_canon and best_sim >= min_accept:
                    self._dbg(f"_find_best_group: synonym fallback '{label}' -> '{best_canon}' (via '{syn_target}', sim={best_sim:.3f})")
                    return best_canon


        # Get embedding for the label (unless in string-only debug mode)
        label_embedding = None
        if not self.debug_string_only:
            label_embedding = self._get_embedding(label)
        if label_embedding is None:
            self._dbg(f"_find_best_group: label embedding unavailable; falling back to string similarity")

        
        # Sort groups by frequency (size) for checking highest frequency first
        sorted_groups = sorted(
            category_groups.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        if self.debug_list_keys:
            keys = [k for k,_ in sorted_groups[:self.debug_list_keys]]
            if self.debug_find_contains:
                keys = [k for k in keys if self.debug_find_contains in k.lower()]
            self._dbg(f"sample canonical keys ({len(keys)}): {keys}")
        
        best_similarity = 0
        best_group = None

        checked = 0
        for canonical, variants in sorted_groups:
            checked += 1
            if self.debug_max_groups and self.debug_max_groups > 0 and checked > self.debug_max_groups:
                self._dbg(f"_find_best_group: abort after {self.debug_max_groups} groups (cap). best={best_group} sim={best_similarity:.3f}")
                break

            # Similarity against a single representative: canonical string
            if self.debug_string_only or label_embedding is None:
                # combine token and trigram similarity
                similarity = max(
                    self._token_jaccard(self._tokens(norm_label), self._tokens(self._normalize_label(canonical))),
                    self._string_similarity_quick(norm_label, self._normalize_label(canonical)),
                )
            else:
                canonical_embedding = self._get_embedding(canonical)
                if canonical_embedding is None:
                    continue
                similarity = self._cosine_similarity(label_embedding, canonical_embedding)

            
            if similarity >= threshold and similarity > best_similarity:
                best_similarity = similarity
                best_group = canonical
                if checked % max(1, self.debug_step) == 0 or self.debug_verbose:
                    self._dbg(f"_find_best_group: new best '{best_group}' @ {best_similarity:.3f} (checked={checked}/{n_groups})")
        
        return best_group

    def _normalize_label(self, s: str) -> str:
        return (s or "").strip().lower().replace("-", "_")

    def _tokens(self, s: str) -> set[str]:
        # split on non-alphanum and underscores; drop empties; simple stem-ish suffix trim
        toks = [t for t in re.split(r"[^a-z0-9]", s) if t]
        # tiny heuristic stems for common action nouns/verbs
        stems = []
        for t in toks:
            if t.endswith("ing") and len(t) > 5: t = t[:-3]
            elif t.endswith("ed") and len(t) > 4: t = t[:-2]
            elif t.endswith("tion") and len(t) > 6: t = t[:-4]
            stems.append(t)
        return set(stems)

    def _token_jaccard(self, A: set[str], B: set[str]) -> float:
        if not A or not B:
            return 0.0
        inter = len(A & B)
        union = len(A | B) or 1
        return inter / union

    def _string_similarity_quick(self, a: str, b: str) -> float:
        """
        Very fast, order-of-magnitude string similarity for debug mode.
        Jaccard on 3-grams; returns 0..1.
        """
        a = (a or "").lower()
        b = (b or "").lower()
        if not a or not b:
            return 0.0
        A = {a[i:i+3] for i in range(max(1, len(a)-2))}
        B = {b[i:i+3] for i in range(max(1, len(b)-2))}
        inter = len(A & B)
        union = len(A | B) or 1
        return inter / union

    def _get_embedding(self, text: str, allow_api: bool = True, *, save: bool = True) -> Optional[np.ndarray]:
        """
        Unified embedding access with persistent cache:
        1) cache → 2) (optional) harmonizer/OpenAI → 3) simulated fallback.
        If allow_api=False, we will NOT call out to harmonizer/OpenAI.
        """
        vec = self.embedding_cache.get(text)
        if vec is not None:
            v = self._fit_dim(np.asarray(vec, dtype=np.float32))
            # upgrade cached vector shape if needed
            if (isinstance(vec, (list, tuple)) and len(vec) != self.embed_dim) or (
                    isinstance(vec, np.ndarray) and vec.shape[0] != self.embed_dim
            ):
                self.embedding_cache[text] = v
                if save and self.embed_cache_save_mode != "off":
                    self._embed_cache_dirty = 1
            return v
        if allow_api:
            # Prefer harmonizer if wired
            try:
                if self.harmonizer is not None and hasattr(self.harmonizer, "get_embedding"):
                    vec = self.harmonizer.get_embedding(text)
            except Exception:
                vec = None
            # Direct OpenAI fallback (no call when string-only debug)
            if vec is None and self.openai_client is not None and not self.debug_string_only:
                try:
                    # unified client wrapper
                    vecs = self.openai_client.embed(text, model=self.embed_model, dim=self.embed_dim)
                    vec = np.asarray(vecs[0], dtype=np.float32)
                except Exception as e:
                    print(f"[warn] embeddings API failed for '{text[:40]}': {e}")
                    vec = None
        # Simulated final fallback (keeps cosine math defined)
        if vec is None and not self.debug_string_only:
            vec = self._simulated_embedding(text)
        if vec is None:
            return None
        vec = self._fit_dim(np.asarray(vec, dtype=np.float32))
        self.embedding_cache[text] = vec
        # Defer or skip disk saves in hot loops
        if save and self.embed_cache_save_mode != "off":
            self._embed_cache_dirty = 1
            now = time.time()
            if (
                    self.embed_cache_save_mode == "immediate"
                    or self._embed_cache_dirty >= self.embed_cache_save_every
                    or (now - self._embed_cache_last_save) >= self.embed_cache_save_secs
            ):
                try:
                    self._save_embedding_cache()
                    self._embed_cache_dirty = 0
                    self._embed_cache_last_save = now
                except Exception as e:
                    print(f"[warn] embedding cache save failed: {e}")
        return vec
    
    
    def _simulated_embedding(self, text: str) -> np.ndarray:
        """Create a simulated embedding based on text features."""
        import hashlib
        dim = int(getattr(self, "embed_dim", 1536))  # use configured dim if present

        features: list[float] = []
        # Length features
        features.append(len(text) / 100.0)
        features.append(len(text.split()) / 20.0)
        # Character distribution
        L = max(len(text), 1)
        for char in "aeiou":
            features.append(text.count(char) / L)
        # Hash-based pseudo-random features (low-cost, deterministic)
        md = hashlib.md5(text.encode("utf-8")).digest()
        for i in range(min(32, len(md))):
            features.append(md[i] / 255.0)
        # Deterministic PRNG tail to reach target dim
        if len(features) < dim:
            seed = int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest()[:8], "big")
            x = seed % 2147483647 or 1
            a, m = 1103515245, 2**31 - 1
            while len(features) < dim:
                x = (a * x + 12345) % m
                # scale to [-1, 1]
                features.append(((x / m) * 2.0) - 1.0)
        v = np.asarray(features[:dim], dtype=np.float32)
        # L2 normalize so cosine works predictably
        n = float(np.linalg.norm(v)) or 1.0
        return (v / n).astype(np.float32)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # --- embedding dimension alignment --------------------------------------
    def _fit_dim(self, vec: np.ndarray) -> np.ndarray:
        """
        Ensure vector is exactly self.embed_dim long.
        - If longer: downsample by mean pooling if divisible, else truncate.
        - If shorter: pad with zeros.
        Returns float32 array.
        """
        v = np.asarray(vec, dtype=np.float32)
        n = v.shape[0]
        d = self.embed_dim
        if n == d:
            return v
        if n > d:
            # mean-pool if cleanly divisible; otherwise truncate
            if n % d == 0:
                r = n // d
                v = v.reshape(d, r).mean(axis=1)
                return v.astype(np.float32, copy=False)
            return v[:d].astype(np.float32, copy=False)
        # n < d → pad zeros
        out = np.zeros(d, dtype=np.float32)
        out[:n] = v
        return out
    
    def _add_to_group(self, label: str, group: str, category: str, tier: str):
        """Add a new label to a harmonization group."""
        groups = self.general_groups if tier == "general" else self.specific_groups
        mapping = self.general_mapping if tier == "general" else self.specific_mapping
        
        if group in groups[category]["groups"]:
            if label not in groups[category]["groups"][group]:
                groups[category]["groups"][group].append(label)
                groups[category]["total_labels"] += 1
        
        mapping[category][label] = group
        
        # Write-through: persist immediately so a fresh full rebuild is unnecessary
        self._maybe_save_harmonization(tier)

    def _maybe_save_harmonization(self, tier: str):
        """Persist the harmonization tier to disk if write-through is enabled."""
        if getattr(self, "write_through_harmonization", True):
            try:
                self._save_harmonization_tier(tier)
            except Exception as e:
                _log(f"[harmonization] failed to save tier={tier}: {e}")

    
    def _save_harmonization_tier(self, tier: str):
        """Save updated harmonization tier."""
        groups = self.general_groups if tier == "general" else self.specific_groups
        file_path = self.harmonization_dir / f"harmonization_{tier}.json"
        
        with open(file_path, 'w') as f:
            json.dump(groups, f, indent=2)
    
    def calculate_similarity_score(
        self,
        current_harmonized: Dict,
        target_harmonized: Dict,
        message_timestamp: datetime
    ) -> float:
        """
        Calculate similarity score between two harmonized label sets.
        
        Args:
            current_harmonized: Harmonized labels for current message
            target_harmonized: Harmonized labels for target message
            message_timestamp: Timestamp of target message for recency
            
        Returns:
            Weighted similarity score
        """
        # Calculate tier-weighted overlap scores
        general_score = self._calculate_tier_overlap(
            current_harmonized["general"],
            target_harmonized["general"]
        )
        
        specific_score = self._calculate_tier_overlap(
            current_harmonized["specific"],
            target_harmonized["specific"]
        )
        
        # Combine with tier weights
        tier_score = (
            general_score * self.general_tier_weight +
            specific_score * self.specific_tier_weight
        )
        
        # Apply recency bias
        days_old = (datetime.now(timezone.utc) - message_timestamp).days
        if days_old > self.recency_cutoff_days:
            return 0.0
        
        recency_weight = self.recency_decay_factor ** days_old
        
        return tier_score * recency_weight
    
    def _calculate_tier_overlap(self, current_tier: Dict, target_tier: Dict) -> float:
        """Calculate overlap score for a single tier."""
        total_score = 0.0
        
        for category in ["topic", "tone", "intent"]:
            current_groups = current_tier.get(category, {})
            target_groups = target_tier.get(category, {})
            
            for group, current_data in current_groups.items():
                if group in target_groups:
                    target_data = target_groups[group]
                    # Multiply the scores (avg_prob * count for each)
                    overlap_score = current_data["score"] * target_data["score"]
                    total_score += overlap_score
        
        return total_score

    def _load_two_tier_maps(self) -> None:
        """
        Build raw->(specific, general) maps from precomputed harmonization files.
        Idempotent and cheap; cache on self.
        """
        if getattr(self, "_canon_map", None) is not None:
            return

        data_dir = getattr(self, "data_dir", Path("data"))
        general_path = Path(data_dir) / "harmonization_general.json"
        specific_path = Path(data_dir) / "harmonization_specific.json"

        with open(general_path, "r") as f:
            general = json.load(f)
        with open(specific_path, "r") as f:
            specific = json.load(f)

        canon_map: Dict[str, Dict[str, Tuple[str, str]]] = {
            "topic": {}, "tone": {}, "intent": {}
        }

        # Invert specific groups first (higher precision)
        for cat in ("topic", "tone", "intent"):
            for canonical, members in specific[cat]["groups"].items():
                for m in members:
                    # default general=canonical; fix below if broader group exists
                    canon_map[cat][m] = (canonical, canonical)

        # Fill/override general group second (broader umbrella)
        for cat in ("topic", "tone", "intent"):
            for canonical, members in general[cat]["groups"].items():
                for m in members:
                    spec, gen = canon_map[cat].get(m, (m, canonical))
                    canon_map[cat][m] = (spec, canonical)

        self._canon_map = canon_map

    def _canonicalize_label_list(self, raw_list, category: str, tier: str = "specific") -> dict[str, float]:
        """
        Robustly canonicalize a possibly-messy label list into {canonical_label: weight}.
        Accepts:
          - [{'label': 'foo', 'p': 0.9}], [{'text': 'foo'}], ['foo', 'bar'], ('foo', 0.7)
          - numpy arrays of strings/objects
        Ignores numeric arrays and empty/None entries.
        """
        out: dict[str, float] = {}
        cmap = self._canon_map.get(category, {})

        # Normalize raw_list into a list of "items" we can read from
        if raw_list is None:
            items = []
        elif isinstance(raw_list, dict):
            # e.g. {'labels': [...]}
            items = raw_list.get("labels") or []
        elif isinstance(raw_list, (list, tuple, set)):
            items = list(raw_list)
        elif isinstance(raw_list, np.ndarray):
            # only treat string/object arrays as label-y
            items = raw_list.tolist() if raw_list.dtype.kind in ("U", "S", "O") else []
        else:
            items = [raw_list]

        for it in items:
            # Coerce each element into a dict-like {label: str, p: float}
            if isinstance(it, str):
                item = {"label": it, "p": 1.0}
            elif isinstance(it, dict):
                item = it
            elif isinstance(it, (list, tuple)) and len(it) >= 1:
                # supports ('foo', 0.8)
                item = {"label": str(it[0]), "p": float(it[1]) if len(it) > 1 else 1.0}
            else:
                continue

            raw = (item.get("label") or item.get("text") or "").strip()
            if not raw:
                continue

            # probability weight with clamping to [0,1]
            try:
                p_val = item.get("p", item.get("probability", 1.0))
                p = float(p_val)
            except Exception:
                p = 1.0
            if p < 0.0:
                p = 0.0
            elif p > 1.0:
                p = 1.0

            # Your existing canonicalization logic (kept intact)
            if raw in cmap:
                spec, gen = cmap[raw]
            else:
                gen = self._find_best_group(raw, category, "general")
                spec = self._find_best_specific_within(raw, category, gen) if gen else None
                if spec is None and gen is not None:
                    spec = raw  # unknown specific under known general
                if gen is None:
                    gen = spec or raw  # self-contained fallback

            key = spec if tier == "specific" else gen
            if key:
                prev = out.get(key)
                out[key] = max(prev, p) if prev is not None else p

        return out

    def _weighted_jaccard(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        """
        Weighted Jaccard over canonical labels.
        Intersection uses min(p), union uses max(p).
        """
        if not a or not b:
            return 0.0
        keys = set(a) | set(b)
        inter = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
        union = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
        return inter / union if union > 0 else 0.0


    def _cat_vec(self, canon_map: Dict[str, float], *, allow_api: bool = False) -> Optional[np.ndarray]:
        """
        Weighted, L2-normalized category vector for a canonical label map.
        Cache-only by default (allow_api=False) so scans never hit network.
        """
        if not canon_map:
            return None
        vecs: list[np.ndarray] = []
        weights: list[float] = []
        for label, w in canon_map.items():
            v = self._get_embedding(label, allow_api=allow_api, save=False)
            if v is None:
                continue
            vecs.append(np.asarray(v, dtype=np.float32))
            weights.append(float(w))
        if not vecs:
            return None
        V = np.vstack(vecs)                         # [m, d]
        W = np.asarray(weights, dtype=np.float32)[:, None]  # [m, 1]
        cat = (V * W).sum(axis=0)                   # [d]
        n = float(np.linalg.norm(cat))
        if n == 0.0:
            return None
        return (cat / n).astype(np.float32)

    def _score_labels(
        self,
        cur: Dict[str, Dict[str, float]],
        tgt: Dict[str, Dict[str, float]],
        *,
        w_topic: float = 0.91,
        w_tone: float = 0.03,
        w_intent: float = 0.06,
        embed_weight: float = 0.0,
        cur_vecs: Optional[tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]] = None,
        allow_api_for_tgt_vecs: bool = False,
    ) -> float:
        """
        Fast, composable similarity between current and target label maps.
        1) Weighted-Jaccard on canonical labels
        2) Optional cached-embedding cosine blend
        Returns a scalar in [0, 1] (not hard-clamped; depends on your weights).
        """
        # label overlap
        s_topic  = self._weighted_jaccard(cur.get("topic", {}),  tgt.get("topic", {}))
        s_tone   = self._weighted_jaccard(cur.get("tone", {}),   tgt.get("tone", {}))
        s_intent = self._weighted_jaccard(cur.get("intent", {}), tgt.get("intent", {}))
        sim = w_topic * s_topic + w_tone * s_tone + w_intent * s_intent

        # embedding blend (cache-only unless you flip allow_api_for_tgt_vecs=True)
        if embed_weight > 0.0 and cur_vecs is not None:
            cv_topic, cv_tone, cv_intent = cur_vecs
            tv_topic  = self._cat_vec(tgt.get("topic",  {}), allow_api=allow_api_for_tgt_vecs)
            tv_tone   = self._cat_vec(tgt.get("tone",   {}), allow_api=allow_api_for_tgt_vecs)
            tv_intent = self._cat_vec(tgt.get("intent", {}), allow_api=allow_api_for_tgt_vecs)
            cos_score = (w_topic * self._cos(cv_topic,  tv_topic) +
                        w_tone  * self._cos(cv_tone,   tv_tone) +
                        w_intent* self._cos(cv_intent, tv_intent))
            sim = (1.0 - embed_weight) * sim + embed_weight * cos_score
        return float(sim)

    def _coerce_ts(self, label_path, label_obj, now_utc: datetime) -> datetime:
        """
        Try, in order:
          1) label_obj timestamp fields (ISO or epoch s/ms)
          2) filename prefix YYYYMMDD
          3) file mtime
          4) fallback: now - 7 days
        Return a timezone-aware UTC datetime.
        """
        # 1) Look for common fields in the label payload
        for key in ("timestamp", "ts", "created_at"):
            ts = label_obj.get(key)
            if ts is None:
                continue
            # epoch seconds or ms
            if isinstance(ts, (int, float)):
                # ms if it looks too big
                if ts > 1e12:
                    ts = ts / 1000.0
                try:
                    return datetime.fromtimestamp(ts, tz=timezone.utc)
                except Exception:
                    pass
            # ISO 8601 strings (including trailing 'Z')
            if isinstance(ts, str):
                try:
                    s = ts.strip()
                    # Accept 'Z' and no-tz forms
                    if s.endswith("Z"):
                        s = s[:-1] + "+00:00"
                    dt = datetime.fromisoformat(s)
                    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                except Exception:
                    pass

        # 2) Try filename prefix YYYYMMDD
        try:
            stem = getattr(label_path, "stem", str(label_path))
            ymd = stem.split("_")[0]
            dt = datetime.strptime(ymd, "%Y%m%d").replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass

        # 3) File mtime (best-effort)
        try:
            mtime = os.path.getmtime(label_path)
            return datetime.fromtimestamp(mtime, tz=timezone.utc)
        except Exception:
            pass

        # 4) Fallback: pretend it's a week old
        from datetime import timedelta
        return now_utc - timedelta(days=7)

    def find_similar_messages(
            self,
            message: str,
            min_chars_recent: Optional[int] = None,
            min_chars_long_term: Optional[int] = None,
    ) -> Tuple[List[Dict], Dict]:
        """
        Build context using precomputed two-tier harmonization.
        No embedding API calls in the scan; optional cosine blend uses cached vectors only.
        """
        # budgets
        min_chars_recent = min_chars_recent or self.context_minimum_char_recent
        min_chars_long_term = min_chars_long_term or self.context_minimum_char_long_term

        # setup optional stall watchdog & memory sampling
        def heartbeat(msg: str = ""):
            if _PROFILE:
                self._soft_watchdog_heartbeat()
        if _PROFILE:
            if _USE_HARD_FAULT:
                faulthandler.enable()
                faulthandler.dump_traceback_later(_STALL_SECS, repeat=False)
            else:
                self._start_soft_watchdog(_STALL_SECS)
            if _MEM_SAMPLING and not tracemalloc.is_tracing():
                tracemalloc.start(25)

        # 0) ensure harmonization maps are loaded
        with PhaseTimer("load_two_tier_maps", self._perf):
            self._load_two_tier_maps()
        now = datetime.now(timezone.utc)


        # Current labels → canonical (specific)
        t1 = time.time()
        with PhaseTimer("label)current", self._perf):
            current_labels = self.get_message_labels(message)
        # Robust summary for debug: count only containers; show scalars as-is
        if self.debug_verbose:
            summary = {}
            for k, v in (current_labels or {}).items():
                if isinstance(v, (list, tuple, set, dict)):
                    summary[k] = len(v)
                else:
                    summary[k] = v
            self._dbg(f"get_message_labels -> {summary} in {time.time() - t1:.3f}s")
        ct0=time.time()
        with PhaseTimer("canonicalize_current", self._perf):
            cur_topics = self._canonicalize_label_list(current_labels.get("topic", []), "topic", tier="specific")
            cur_tones = self._canonicalize_label_list(current_labels.get("tone", []), "tone", tier="specific")
            cur_intents = self._canonicalize_label_list(current_labels.get("intent", []), "intent", tier="specific")

        # Create files for current message
        chunk_file, label_file = self._create_message_files(message, current_labels)
        try:
            with open(label_file, "r", encoding="utf-8") as _f:
                _new_gid = (json.load(_f) or {}).get("gid")
        except Exception:
            _new_gid = None

        # General-tier sets for quick overlap gate
        cur_gen_topic = set(
            self._canonicalize_label_list(current_labels.get("topic", []), "topic", tier="general").keys())
        cur_gen_tone = set(self._canonicalize_label_list(current_labels.get("tone", []), "tone", tier="general").keys())
        cur_gen_intent = set(
            self._canonicalize_label_list(current_labels.get("intent", []), "intent", tier="general").keys())

        # IMPORTANT: no embedding blend during the main scan by default
        use_embed_blend = bool(getattr(self, "use_embedding_blend", True)) and hasattr(self, "embedding_cache")
        _env_weight = float(getattr(self, "embed_blend_weight", 0.6))
        embed_weight = (_env_weight if (use_embed_blend and self.embed_in_scan) else 0.0)

        # Precompute current vectors once (only if blending)
        _cur_v_topic = self._cat_vec(cur_topics, allow_api=False)
        _cur_v_tone = self._cat_vec(cur_tones, allow_api=False)
        _cur_v_intent = self._cat_vec(cur_intents, allow_api=False)
        _cur_vecs = (_cur_v_topic, _cur_v_tone, _cur_v_intent)

        # ---------- Candidate scan over label files ----------
        similarities: List[Dict] = []
        label_files = list(self.labels_dir.glob("*.json"))  # list once so we can show progress
        n_labels = len(label_files)
        self._dbg(f"begin candidate scan (files={n_labels})")

        with PhaseTimer("scan_candidate_labels", self._perf):
            block_t0 = time.perf_counter()
            for i, label_file_path in enumerate(label_files, 1):
                # Progress heartbeat every N files
                if _PROFILE and (i % _LOG_EVERY == 0 or i == n_labels):
                    dt = time.perf_counter() - block_t0
                    _log(f"scan {i}/{n_labels} (~{100 * i // max(1, n_labels)}%) sims={len(similarities)} in {dt:.2f}s")
                    heartbeat(f"scan i={i}")

                try:
                    with open(label_file_path, "r", encoding="utf-8") as f:
                        target_labels = json.load(f)
                    gid = target_labels.get("gid")
                    if not gid: continue
                    if _new_gid and gid == _new_gid:  # skip the just-created one
                        continue

                    # Timestamp
                    ts_str = label_file_path.stem.split("_")[0]
                    try:
                        ts = datetime.strptime(ts_str, "%Y%m%d").replace(tzinfo=timezone.utc)
                    except Exception:
                        ts = target_labels.get("timestamp")
                        if isinstance(ts, str):
                            try:
                                ts = datetime.fromisoformat(ts)
                            except Exception:
                                ts = None
                        if not isinstance(ts, datetime):
                            ts = self._coerce_ts(label_file_path, target_labels, now)

                    # General-tier overlap gate
                    tgt_gen_topic = set(
                        self._canonicalize_label_list(target_labels.get("topic", []), "topic", tier="general").keys())
                    tgt_gen_tone = set(
                        self._canonicalize_label_list(target_labels.get("tone", []), "tone", tier="general").keys())
                    tgt_gen_intent = set(
                        self._canonicalize_label_list(target_labels.get("intent", []), "intent", tier="general").keys())
                    if not (
                            cur_gen_topic & tgt_gen_topic or cur_gen_tone & tgt_gen_tone or cur_gen_intent & tgt_gen_intent):
                        continue

                    # Target canonical (specific)
                    tgt_topics = self._canonicalize_label_list(target_labels.get("topic", []), "topic", tier="specific")
                    tgt_tones = self._canonicalize_label_list(target_labels.get("tone", []), "tone", tier="specific")
                    tgt_intents = self._canonicalize_label_list(target_labels.get("intent", []), "intent",
                                                                tier="specific")

                    # One call computes label-overlap (and optional cached-embedding blend)
                    w_topic, w_tone, w_intent = 0.91, 0.03, 0.06
                    sim = self._score_labels(
                        {"topic": cur_topics, "tone": cur_tones, "intent": cur_intents},
                        {"topic": tgt_topics, "tone": tgt_tones, "intent": tgt_intents},
                        w_topic=w_topic, w_tone=w_tone, w_intent=w_intent,
                        embed_weight=embed_weight,          # 0.0 in scan unless BB_EMBED_IN_SCAN=1
                        cur_vecs=None if embed_weight == 0.0 else _cur_vecs,
                        allow_api_for_tgt_vecs=False,       # cache-only even if enabled
                    )

                    if sim > 0.2:
                        similarities.append({
                            "gid": gid,
                            "similarity": float(sim),
                            "timestamp": ts,
                            "label_file": label_file_path.name,
                        })
                        if getattr(self, "debug_similarity", False):
                            print(f"[ctx] {gid} sim={sim:.4f}")

                except Exception as e:
                    print(f"[ctx] Error processing {label_file_path}: {e}")
                    continue

        self._dbg("end candidate scan")
        if _PROFILE:
            _log(f"similarities: {len(similarities)}")

        # ---------- Sort & assemble context ----------
        with PhaseTimer("sort_similarities", self._perf):
            similarities.sort(key=lambda x: x["similarity"], reverse=True)

        # Optional post-scan embedding rerank on top-K (fetch embeddings first)
        if self.use_embedding_blend and self.embed_blend_weight > 0.0 and similarities:
            t_r0 = time.time()
            K = self.embed_rerank_topk if self.embed_rerank_topk > 0 else len(similarities)
            head = similarities[:K]
            self._dbg(f"rerank start: topK={K}, blend_w={self.embed_blend_weight}, api={self.embed_rerank_allow_api}")
            # Collect all canonical labels we need embeddings for (current  targets)
            needed = set(cur_topics.keys()) | set(cur_tones.keys()) | set(cur_intents.keys())
            tgt_canon_cache: dict[str, tuple[dict[str, float], dict[str, float], dict[str, float]]] = {}
            for item in head:
                p = self.labels_dir / item["label_file"]
                tgt = {}
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        tgt = json.load(f)
                    tt = self._canonicalize_label_list(tgt.get("topic"), "topic", tier="specific")
                    to = self._canonicalize_label_list(tgt.get("tone"), "tone", tier="specific")
                    ti = self._canonicalize_label_list(tgt.get("intent"), "intent", tier="specific")
                    tgt_canon_cache[item["label_file"]] = (tt, to, ti)
                    # add targets to 'needed' so _ensure_embeddings_for can warm the cache
                    needed |= set(tt.keys()) | set(to.keys()) | set(ti.keys())
                except Exception as e:
                    shapes = {k: type(v).__name__ for k, v in (tgt.items() if isinstance(tgt, dict) else []) if
                              k in ("topic", "tone", "intent")}
                    _log(f"[ctx] Error processing {p.name}: {e} | shapes={shapes}")
                    continue
            # Warm cache ONLY from local cache or simulated (no API by default)
            self._ensure_embeddings_for(sorted(needed), allow_api=self.embed_rerank_allow_api)
            # Build current category vectors once (helpers)
            cur_v_topic  = self._cat_vec(cur_topics,  allow_api=self.embed_rerank_allow_api)
            cur_v_tone   = self._cat_vec(cur_tones,   allow_api=self.embed_rerank_allow_api)
            cur_v_intent = self._cat_vec(cur_intents, allow_api=self.embed_rerank_allow_api)

            reranked = []
            for item in head:
                tt, to, ti = tgt_canon_cache.get(item["label_file"], ({}, {}, {}))
                tvt = self._cat_vec(tt, allow_api=self.embed_rerank_allow_api)
                tvo = self._cat_vec(to, allow_api=self.embed_rerank_allow_api)
                tvi = self._cat_vec(ti, allow_api=self.embed_rerank_allow_api)
                cos_score = (0.8 * self._cos(cur_v_topic,  tvt) +
                             0.1 * self._cos(cur_v_tone,   tvo) +
                             0.1 * self._cos(cur_v_intent, tvi))
                new_sim = (1.0 - self.embed_blend_weight) * item["similarity"] + self.embed_blend_weight * cos_score
                it2 = dict(item)
                it2["similarity"] = float(new_sim)
                reranked.append(it2)
            similarities[:len(reranked)] = sorted(reranked, key=lambda x: x["similarity"], reverse=True)
            self._dbg(f"rerank done in {time.time() - t_r0:.2f}s")

        #ensure we can resolve gid -> chunk file quickly
        with PhaseTimer("build_chunk_index", self._perf):
            self._ensure_chunk_index()

        context_messages: List[Dict] = []
        recent_chars = 0
        long_chars = 0

        with PhaseTimer("assemble_context", self._perf):
            block_t0 = time.perf_counter()
            total = len(similarities)
            debug_info: list[dict] = []
            include_gids: list[str] = []
            for j, item in enumerate(similarities):
                # I/O timing around chunk load
                t_io0 = time.perf_counter()
                try:
                    chunk_content = self._load_chunk_by_gid(item["gid"]) or {}
                except json.JSONDecodeError as e:
                    _log(f"[assemble] JSON decode failed gid={item['gid']} -> {e}")
                    continue
                except Exception as e:
                    _log(f"[assemble] load_chunk_by_gid({item['gid']}) error: {e}")
                    continue
                dt_io = time.perf_counter() - t_io0
                self._metrics.io_calls += 1
                self._metrics.io_time += dt_io
                if _PROFILE and dt_io > _IO_WARN_SECS:
                    _log(f"[assemble] slow chunk gid={item['gid']} took {dt_io:.2f}s")
                text = (chunk_content.get("content_text") or "")
                L = len(text)
                is_recent = (now - item["timestamp"]).days <= 7

                if is_recent:
                    if recent_chars + L > min_chars_recent:
                        # still log progress so we can see it moving
                        pass
                    else:
                        recent_chars += L
                else:
                    if long_chars + L > min_chars_long_term:
                        pass
                    else:
                        long_chars += L

                if ((is_recent and recent_chars <= min_chars_recent) or
                        (not is_recent and long_chars <= min_chars_long_term)):
                    context_messages.append({
                        "gid": item["gid"],
                        "content": text,
                        "similarity": item["similarity"],
                        "timestamp": item["timestamp"].isoformat(),
                        "is_recent": is_recent,
                    })

                    include_gids.append(item["gid"])

                # record a compact debug row for this candidate
                debug_info.append({
                    "gid": item["gid"],
                    "label_file": item.get("label_file"),
                    "similarity": float(item["similarity"]),
                    "chunk_found": bool(chunk_content),
                    "chunk_path": chunk_content.get("__path"),
                    "content_len": L,
                    "used_field": (chunk_content.get("__extract_diag") or {}).get("used_field"),
                    "raw_parse": (chunk_content.get("__extract_diag") or {}).get("raw_parse"),
                    "keys": [k for k in chunk_content.keys() if not k.startswith("__")][:12],
                })

                # progress ping
                if _PROFILE and (j % _LOG_EVERY == 0 or j == total):
                    dt = time.perf_counter() - block_t0
                    _log(f"assemble {j}/{total} added={len(context_messages)} "
                         f"recent={recent_chars}/{min_chars_recent} long={long_chars}/{min_chars_long_term} in {dt:.2f}s")
                    heartbeat(f"assemble")

                if recent_chars >= min_chars_recent and long_chars >= min_chars_long_term:
                    break

        metadata = {
            "current_labels": current_labels,
            "current_canonical": {"topic": cur_topics, "tone": cur_tones, "intent": cur_intents},
            "total_similarities_found": len(similarities),
            "context_messages_included": len(context_messages),
            "recent_chars": recent_chars,
            "long_term_chars": long_chars,
            "chunk_file": chunk_file,
            "label_file": label_file,
            "stopped_due_to": "char_budget" if (
                    recent_chars >= min_chars_recent and long_chars >= min_chars_long_term
            ) else "exhausted_candidates",
            # NEW: surface GIDs + compact per-candidate diagnostics
            "similar_message_gids": include_gids,
            "similarity_debug": debug_info[:int(os.getenv("BB_MATCHER_DEBUG_LIMIT", "200"))],
        }

        if _PROFILE:
            total_time = sum(self._perf.values())
            _log("PHASES: " + " | ".join(f"{k}={v:.3f}s" for k, v in self._perf.items()))
            _log(f"TOTAL: {total_time:.3f}s | io_calls={self._metrics.io_calls} io_time={self._metrics.io_time:.3f}s")
            if _MEM_SAMPLING and tracemalloc.is_tracing():
                cur, peak = tracemalloc.get_traced_memory()
                _log(f"mem: current={cur / 1e6:.1f}MB peak={peak / 1e6:.1f}MB")
                tracemalloc.stop()
            if _USE_HARD_FAULT:
                faulthandler.cancel_dump_traceback_later()
            else:
                self._stop_soft_watchdog()

        return context_messages, metadata

    def _cos(self, a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        if a is None or b is None:
            return 0.0
        da = float(np.linalg.norm(a)); db = float(np.linalg.norm(b))
        if da == 0.0 or db == 0.0:
            return 0.0
        return float(np.dot(a, b) / (da * db))

    def _create_message_files(self, message: str, labels: Dict) -> Tuple[str, str]:
        """
        Create chunk and label files for a new message.
        
        Args:
            message: The message content
            labels: Generated labels
            
        Returns:
            Tuple of (chunk_filename, label_filename)
        """
        # Generate unique ID
        timestamp = datetime.now(timezone.utc)
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        msg_hash = hashlib.md5(message.encode()).hexdigest()[:8]
        gid = f"msg_{timestamp_str}_{msg_hash}"
        
        # Create chunk file
        chunk_data = {
            "gid": gid,
            "role": "user",
            "content_text": message,
            "ts": timestamp.isoformat(),
            "model": None
        }
        
        chunk_file = self.chunks_dir / f"{timestamp_str}_{msg_hash}.json"
        with open(chunk_file, 'w') as f:
            json.dump(chunk_data, f, indent=2)
        
        # Create label file
        label_data = {
            "gid": gid,
            "topic": labels.get("topic", []),
            "tone": labels.get("tone", []),
            "intent": labels.get("intent", []),
            "confidence": labels.get("confidence", 0.5)
        }
        
        label_file = self.labels_dir / f"{timestamp_str}_{msg_hash}.json"
        with open(label_file, 'w') as f:
            json.dump(label_data, f, indent=2)
        
        return chunk_file.name, label_file.name

    # ---------- Text extraction helpers ----------
    def _coalesce_text_from_content(self, value):
        """
        Normalize text from common content shapes:
          - str
          - list of blocks: [{'type':'text','text': '...'}, ...]
          - dict with 'parts': {'parts': ['a','b',...]} (legacy ChatML-ish)
          - dict with 'text' or 'content' nesting
        Returns a single string.
        """
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            out = []
            for item in value:
                if isinstance(item, dict):
                    # Prefer modern OpenAI content blocks
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        out.append(item["text"])
                    else:
                        # Fallbacks
                        t = item.get("text") or item.get("content")
                        if isinstance(t, list):
                            out += [str(x) for x in t]
                        elif isinstance(t, str):
                            out.append(t)
                elif isinstance(item, (str, int, float)):
                    out.append(str(item))
            return "\n".join(x for x in out if x)
        if isinstance(value, dict):
            # Legacy ChatML-ish: {'parts': [...]}
            if "parts" in value and isinstance(value["parts"], list):
                return "\n".join(str(x) for x in value["parts"] if isinstance(x, (str, int, float)))
            inner = value.get("text") or value.get("content") or value.get("message")
            return self._coalesce_text_from_content(inner)
        return ""

    def _extract_text_with_diag(self, obj: dict) -> tuple[str, dict]:
        """
        Like _extract_text_from_chunk, but returns (text, diagnostics).
        Diagnostics include which field/path was used and parse status for 'raw'.
        """
        diag = {
            "used_field": None,
            "raw_present": bool(obj.get("raw") or obj.get("raw_meta") or obj.get("raw_message")),
            "raw_parse": "n/a",
            "len": 0
        }
        # 1) Flat/near-flat fields
        for key in ("content_text", "content", "text", "message"):
            if key in obj and obj[key]:
                t = self._coalesce_text_from_content(obj[key])
                if t:
                    diag["used_field"] = key
                    diag["len"] = len(t)
                    return t, diag

        # 2) Raw envelope(s)
        raw = obj.get("raw") or obj.get("raw_meta") or obj.get("raw_message")
        if raw:
            try:
                raw_obj = json.loads(raw) if isinstance(raw, str) else raw
                diag["raw_parse"] = "ok"
            except Exception:
                diag["raw_parse"] = "json_error"
                return "", diag
            if isinstance(raw_obj, dict):
                for path in (("content",),
                             ("message", "content"),
                             ("data", "message", "content")):
                    cur = raw_obj
                    for p in path:
                        if isinstance(cur, dict) and p in cur:
                            cur = cur[p]
                        else:
                            cur = None
                            break
                    if cur is not None:
                        t = self._coalesce_text_from_content(cur)
                        if t:
                            diag["used_field"] = "raw:" + ".".join(path)
                            diag["len"] = len(t)
                            return t, diag
        return "", diag

    def _extract_text_from_chunk(self, obj: dict) -> str:
        """Best-effort extraction of human text from a chunk record.
        Handles:
          - flat fields: content_text, content, text, message
          - OpenAI messages: {"content":[{"type":"text","text":"..."}]}
          - legacy ChatML: {"content":{"content_type":"text","parts":["..."]}}
          - objects under 'raw' (stringified JSON or nested dict)
        """

        def _from_content(value):
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                parts = []
                for item in value:
                    if isinstance(item, dict):
                        t = item.get("text") or item.get("content") or ""
                        if isinstance(t, list):
                            parts.extend([str(x) for x in t])
                        elif isinstance(t, str):
                            parts.append(t)
                return "\n".join([p for p in parts if p])
            if isinstance(value, dict):
                if "parts" in value and isinstance(value["parts"], list):
                    return "\n".join([str(x) for x in value["parts"] if isinstance(x, (str, int, float))])
                inner = value.get("text") or value.get("content")
                return _from_content(inner)
            return ""

        for key in ("content_text", "content", "text", "message"):
            v = obj.get(key)
            if v:
                t = _from_content(v)
                if t:
                    return t

        raw = obj.get("raw") or obj.get("raw_meta") or obj.get("raw_message")
        if raw:
            try:
                raw_obj = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                raw_obj = None
            if isinstance(raw_obj, dict):
                for path in (("content",), ("message", "content"), ("data", "message", "content")):
                    cur = raw_obj
                    for p in path:
                        if isinstance(cur, dict) and p in cur:
                            cur = cur[p]
                        else:
                            cur = None
                            break
                    if cur is not None:
                        t = _from_content(cur)
                        if t:
                            return t
        return ""

    def _load_chunk_by_gid(self, gid: str) -> Optional[Dict]:
        """Load chunk content by gid using an index and a small in-memory cache."""
        # cache hit
        hit = self._chunk_cache.get(gid)
        if hit is not None:
            return hit
        # index lookup
        if self._chunk_index is None:
            self._ensure_chunk_index()
        path = self._chunk_index.get(gid) if self._chunk_index else None
        diag = ""

        # Fallbacks: try direct filenames if not indexed
        if not path:
            candidates = [self.chunks_dir / f"{gid}.json"]
            if gid.startswith("msg_"):
                candidates.append(self.chunks_dir / (gid[4:] + ".json"))
            extra = os.getenv("BB_MATCHER_EXTRA_CHUNK_DIRS", "")
            for base in [Path(p) for p in re.split(r"[;:]", extra) if p.strip()]:
                candidates.append(Path(base) / f"{gid}.json")
                if gid.startswith("msg_"):
                    candidates.append(Path(base) / (gid[4:] + ".json"))

            for cand in candidates:
                try:
                    if cand.exists():
                        with open(cand, "r", encoding="utf-8-sig") as f:
                            obj = json.load(f)
                        obj.setdefault("gid", gid)
                        obj["__path"] = str(cand)
                        # Ensure content_text is populated
                        if not obj.get("content_text"):
                            # use diagnostic extractor if present
                            if hasattr(self, "_extract_text_with_diag"):
                                txt, diag = self._extract_text_with_diag(obj)
                                obj["content_text"] = txt or ""
                                obj["__extract_diag"] = diag
                            else:
                                obj["content_text"] = self._extract_text_from_chunk(obj)
                        self._chunk_cache[gid] = obj
                        return obj
                except Exception:
                    pass
            return None

        with open(path, "r", encoding="utf-8-sig") as f:
            obj = json.load(f)
        obj["__path"] = str(path)
        if not obj.get("content_text"):
            if hasattr(self, "_extract_text_with_diag"):
                txt, diag = self._extract_text_with_diag(obj)
                obj["content_text"] = txt or ""
                obj["__extract_diag"] = diag
            else:
                obj["content_text"] = self._extract_text_from_chunk(obj)
            obj["__extract_diag"] = diag
        else:
            obj["__extract_diag"] = {"used_field": "content_text", "raw_present": bool(obj.get("raw")),
                                     "raw_parse": "n/a", "len": len(obj.get("content_text", ""))}
        # tiny LRU-ish cap
        self._chunk_cache[gid] = obj
        if len(self._chunk_cache) > int(os.getenv("BB_MATCHER_CHUNK_CACHE", "2048")):
            # pop an arbitrary item (Python 3.7 dicts are insertion-ordered; this pops oldest)
            self._chunk_cache.pop(next(iter(self._chunk_cache)))
        return obj
    
    def build_context_header(
        self,
        message: str,
        max_chars: int = 150000
    ) -> str:
        """
        Build a context header for a message.
        
        Args:
            message: The current message
            max_chars: Maximum characters in context
            
        Returns:
            Formatted context header string
        """
        context_messages, metadata = self.find_similar_messages(message)
        
        # Build the header
        header_parts = []
        
        # Add metadata
        header_parts.append(f"# Context Header")
        header_parts.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        header_parts.append(f"Similar messages found: {metadata['total_similarities_found']}")
        header_parts.append(f"Context messages included: {metadata['context_messages_included']}")
        header_parts.append("")
        
        # Add harmonized labels summary
        header_parts.append("## Current Message Labels")
        for tier in ["general", "specific"]:
            header_parts.append(f"\n### {tier.capitalize()} Tier")
            for category in ["topic", "tone", "intent"]:
                groups = metadata["current_harmonized"][tier][category]
                if groups:
                    top_groups = sorted(
                        groups.items(),
                        key=lambda x: x[1]["score"],
                        reverse=True
                    )[:3]
                    header_parts.append(f"**{category}**: {', '.join(g[0] for g in top_groups)}")
        
        header_parts.append("")
        header_parts.append("## Similar Context Messages")
        
        # Add context messages
        for i, msg in enumerate(context_messages, 1):
            header_parts.append(f"\n### Message {i} (similarity: {msg['similarity']:.3f})")
            header_parts.append(f"Timestamp: {msg['timestamp']}")
            header_parts.append(f"Recent: {msg['is_recent']}")
            header_parts.append("```")
            header_parts.append(msg['content'][:500])  # Truncate long messages
            if len(msg['content']) > 500:
                header_parts.append("...")
            header_parts.append("```")
        
        return "\n".join(header_parts)


# Example usage
if __name__ == "__main__":
    from app.openai_client import OpenAIClient
    
    # Initialize
    client = OpenAIClient()
    matcher = ContextSimilarityMatcher(
        labels_dir="labels",
        chunks_dir="data/chunks",
        harmonization_dir="data",
        openai_client=client,
        context_minimum_char_long_term=50000,
        context_minimum_char_recent=10000
    )
    
    # Test message
    test_message = "I took my vitamins this morning and went for a 30 minute run in the park."
    
    # Find similar messages
    context_messages, metadata = matcher.find_similar_messages(test_message)
    
    print(f"Found {len(context_messages)} similar messages")
    print(f"Recent context: {metadata['recent_chars']} chars")
    print(f"Long-term context: {metadata['long_term_chars']} chars")
    
    # Build context header
    header = matcher.build_context_header(test_message)
    print("\nContext Header:")
    print(header[:1000])  # Print first 1000 chars