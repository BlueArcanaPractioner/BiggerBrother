"""
Context Similarity Matcher for BiggerBrother
============================================

Implements weighted similarity matching between messages using harmonized labels,
tier weights, and recency bias to build optimal context headers.
"""

from __future__ import annotations
import json
import os
import math
import time
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
import re
import tempfile, pickle

from app.openai_client import OpenAIClient
from app.label_integration_wrappers import LabelGenerator
from assistant.importers.enhanced_smart_label_importer import EnhancedLabelHarmonizer



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
        self.labels_dir = Path(labels_dir)
        self.chunks_dir = Path(chunks_dir)
        self.harmonization_dir = Path(harmonization_dir)
        
        self.openai_client = openai_client or OpenAIClient()
        self.label_generator = LabelGenerator(self.openai_client)
        self.harmonizer = harmonizer

        # Embedding cache file: unify with harmonizer if available
        if self.harmonizer and hasattr(self.harmonizer, "embedding_cache_file"):
            self.embedding_cache_file = self.harmonization_dir / 'embedding_cache.pkl'
        
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
                os.fsync(f.fileno())
            os.replace(tmp_path, cache_file)  # atomic on same filesystem
        finally:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
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
        
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Return embedding for 'test', populating and PERSISTING the cache on miss.
        """

        vec = self.embedding_cache.get(text)
        if vec is not None:
            return vec
        vec = self.harmonizer.get_embedding(text)
        if vec is None:
            return None
        self.embedding_cache[text] = vec

        try:
            self._save_embedding_cache()
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.warning(f"embedding cache save failed: {e}")
            else:
                print(f"[warn] embedding cache save failed: {e}")
        return vec
    
    
    def _simulated_embedding(self, text: str) -> np.ndarray:
    
        """Create a simulated embedding based on text features."""
    
        import hashlib
    
        features = []
    
        
    
        # Length features
    
        features.append(len(text) / 100)
    
        features.append(len(text.split()) / 20)
    
        
    
        # Character distribution
    
        for char in 'aeiou':
    
            features.append(text.count(char) / max(len(text), 1))
    
        
    
        # Hash-based pseudo-random features
    
        text_hash = hashlib.md5(text.encode()).digest()
    
        for i in range(10):
    
            features.append(text_hash[i] / 255)
    
        
    
        # Pad to standard embedding size (1536 for OpenAI)
    
        while len(features) < 1536:
    
            features.append(0)
    
        
    
        return np.array(features[:1536])
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _add_to_group(self, label: str, group: str, category: str, tier: str):
        """Add a new label to a harmonization group."""
        groups = self.general_groups if tier == "general" else self.specific_groups
        mapping = self.general_mapping if tier == "general" else self.specific_mapping
        
        if group in groups[category]["groups"]:
            if label not in groups[category]["groups"][group]:
                groups[category]["groups"][group].append(label)
                groups[category]["total_labels"] += 1
        
        mapping[category][label] = group
        
        # Save updated harmonization
        self._save_harmonization_tier(tier)
    
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
        out: dict[str, float] = {}
        cmap = self._canon_map.get(category, {})

        for item in (raw_list or []):
            raw = item.get("label") or item.get("text") or ""
            if not raw:
                continue
            p = float(item.get("p", item.get("probability", 1.0)))
            p = 0.0 if p < 0 else (1.0 if p > 1.0 else p)

            if raw in cmap:
                spec, gen = cmap[raw]
            else:
                # reuse your helper(s), no new logic
                gen = self._find_best_group(raw, category, "general")
                spec = self._find_best_specific_within(raw, category, gen) if gen else None
                # fallbacks that keep us moving
                if spec is None and gen is not None:
                    spec = raw  # unknown specific under known general
                if gen is None:
                    gen = spec or raw  # keep it self-contained

            key = spec if tier == "specific" else gen
            if key:
                out[key] = max(out.get(key, 0.0), p)

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

        # Setup
        t0 = time.time()
        self._dbg("begin finding similar messages")
        self._load_two_tier_maps()
        self._dbg(f"loaded two-tier maps in {time.time() - t0:.3f}s")
        now = datetime.now(timezone.utc)

        # Current labels → canonical (specific)
        t1 = time.time()
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
        cur_topics = self._canonicalize_label_list(current_labels.get("topic", []), "topic", tier="specific")
        cur_tones = self._canonicalize_label_list(current_labels.get("tone", []), "tone", tier="specific")
        cur_intents = self._canonicalize_label_list(current_labels.get("intent", []), "intent", tier="specific")
        self._dbg(f"canonicalized: topics={len(cur_topics)}, tones={len(cur_tones)}, intents={len(cur_intents)} in {time.time()-t0:.3f}s")

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

        # Optional cosine blend from cached embeddings (no API)
        use_embed_blend = getattr(self, "use_embedding_blend", False) and hasattr(self, "embedding_cache")
        embed_weight = float(getattr(self, "embed_blend_weight", 0.6))
        embed_weight = 0.0 if not use_embed_blend else max(0.0, min(1.0, embed_weight))

        def _cat_vec(canon: Dict[str, float]) -> Optional[List[float]]:
            if not use_embed_blend:
                return None
            acc = None
            for lab, w in canon.items():
                v = self.embedding_cache.get(lab)
                if v is None:
                    continue
                acc = [w * x for x in v] if acc is None else [a + w * x for a, x in zip(acc, v)]
            if acc is None:
                return None
            import math
            n = math.sqrt(sum(a * a for a in acc)) or 0.0
            return [a / n for a in acc] if n > 0 else None

        def _cos(u: Optional[List[float]], v: Optional[List[float]]) -> float:
            if u is None or v is None:
                return 0.0
            return float(sum(a * b for a, b in zip(u, v)))

        # Precompute current vectors once (only if blending)
        _cur_v_topic = _cat_vec(cur_topics)
        _cur_v_tone = _cat_vec(cur_tones)
        _cur_v_intent = _cat_vec(cur_intents)

        self._dbg("begin candidate scan")
        similarities: List[Dict] = []
        for label_file_path in self.labels_dir.glob("*.json"):
            try:
                with open(label_file_path, "r", encoding="utf-8") as f:
                    target_labels = json.load(f)
                gid = target_labels.get("gid")
                if not gid:
                    continue
                if _new_gid and gid == _new_gid:
                    continue

                # Timestamp: try filename, then label fields, then file mtime, else ~7d ago
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
                tgt_intents = self._canonicalize_label_list(target_labels.get("intent", []), "intent", tier="specific")

                # Weighted Jaccard
                w_topic, w_tone, w_intent = 0.5, 0.2, 0.3
                s_topic = self._weighted_jaccard(cur_topics, tgt_topics)
                s_tone = self._weighted_jaccard(cur_tones, tgt_tones)
                s_intent = self._weighted_jaccard(cur_intents, tgt_intents)
                sim = w_topic * s_topic + w_tone * s_tone + w_intent * s_intent

                # Optional cached-embedding cosine blend (applied once)
                if embed_weight > 0.0 and (_cur_v_topic or _cur_v_tone or _cur_v_intent):
                    tgt_v_topic = _cat_vec(tgt_topics)
                    tgt_v_tone = _cat_vec(tgt_tones)
                    tgt_v_intent = _cat_vec(tgt_intents)
                    cos_score = (
                            w_topic * _cos(_cur_v_topic, tgt_v_topic) +
                            w_tone * _cos(_cur_v_tone, tgt_v_tone) +
                            w_intent * _cos(_cur_v_intent, tgt_v_intent)
                    )
                    sim = (1.0 - embed_weight) * sim + embed_weight * cos_score

                if sim > 0.0:
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
        # Sort & build context within budgets
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        context_messages: List[Dict] = []
        recent_chars = 0
        long_chars = 0

        for item in similarities:
            chunk_content = self._load_chunk_by_gid(item["gid"]) or {}
            text = chunk_content.get("content_text", "") or ""
            L = len(text)
            is_recent = (now - item["timestamp"]).days <= 7

            if is_recent:
                if recent_chars + L > min_chars_recent:
                    continue
                recent_chars += L
            else:
                if long_chars + L > min_chars_long_term:
                    continue
                long_chars += L

            context_messages.append({
                "gid": item["gid"],
                "content": text,
                "similarity": item["similarity"],
                "timestamp": item["timestamp"].isoformat(),
                "is_recent": is_recent,
            })

            if recent_chars >= min_chars_recent and long_chars >= min_chars_long_term:
                break

        metadata = {
            "current_labels": current_labels,
            "current_canonical": {
                "topic": cur_topics, "tone": cur_tones, "intent": cur_intents
            },
            "total_similarities_found": len(similarities),
            "context_messages_included": len(context_messages),
            "recent_chars": recent_chars,
            "long_term_chars": long_chars,
            "chunk_file": chunk_file,
            "label_file": label_file,
            "stopped_due_to": "char_budget" if (
                    recent_chars >= min_chars_recent and long_chars >= min_chars_long_term
            ) else "exhausted_candidates",
        }
        return context_messages, metadata

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
    
    def _load_chunk_by_gid(self, gid: str) -> Optional[Dict]:
        """Load chunk content by gid."""
        # Search for chunk with matching gid
        for chunk_file in self.chunks_dir.glob("*.json"):
            try:
                with open(chunk_file, 'r') as f:
                    chunk = json.load(f)
                    if chunk.get("gid") == gid:
                        return chunk
            except:
                continue
        
        return None
    
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