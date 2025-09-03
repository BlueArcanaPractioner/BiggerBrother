"""
Enhanced Smart Label Importer with Two-Tier Harmonization
===========================================================

Uses two-tier similarity thresholds without fixed group targets.
Combines string matching and embeddings to minimize API calls.
"""

import json
import os
import time

import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional: Only import OpenAI if available
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.info("OpenAI library not installed. Will use simulated embeddings.")

from assistant.graph.store import GraphStore


def ensure_timezone_aware(dt: Optional[datetime] = None) -> datetime:
    """Ensure datetime is timezone-aware. Use UTC if no timezone specified."""
    if dt is None:
        return datetime.now(timezone.utc)
    if not hasattr(dt, 'tzinfo') or dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class EnhancedLabelHarmonizer:
    """
    Enhanced harmonizer with two-tier similarity thresholds.
    Uses smart filtering to minimize embedding API calls.
    No fixed group targets - groups emerge naturally from data.
    """

    def __init__(
        self,
        harmonization_report_path: str = "data/harmonization_report.json",
        target_groups: Dict[str, int] = None,  # Kept for compatibility but ignored
        similarity_threshold: float = 0.80,  # Specific threshold
        min_semantic_distance: float = 0.5,  # General threshold
        use_real_embeddings: bool = True,
        embedding_cache_file: str = "data/embedding_cache.pkl"
    ):
        self.harmonization_report_path = Path(harmonization_report_path)
        self.data_dir = self.harmonization_report_path.parent

        # Compatibility attributes (kept but not used for grouping)
        self.target_groups = target_groups or {}
        self.similarity_threshold = similarity_threshold
        self.min_semantic_distance = min_semantic_distance

        # Two-tier thresholds
        self.general_threshold = min_semantic_distance  # 0.5 for general groups
        self.specific_threshold = similarity_threshold  # 0.8 for specific groups

        self.use_real_embeddings = use_real_embeddings and OPENAI_AVAILABLE
        self.embedding_cache_file = Path(embedding_cache_file)

        # Paths for the two harmonization files
        self.general_harmony_file = self.data_dir / "harmonization_general.json"
        self.specific_harmony_file = self.data_dir / "harmonization_specific.json"

        # Two-tier data structures
        self.general_groups = {
            'topic': {},
            'tone': {},
            'intent': {}
        }
        self.specific_groups = {
            'topic': {},
            'tone': {},
            'intent': {}
        }

        # Core structures
        self.label_mappings = {
            'topic': {},
            'tone': {},
            'intent': {}
        }
        self.label_frequencies = {
            'topic': Counter(),
            'tone': Counter(),
            'intent': Counter()
        }
        self.embedding_cache = {}
        self.co_occurrence = defaultdict(lambda: defaultdict(Counter))
        # Similarity cache for batch mode
        self.similarity_cache = {}

        # OpenAI client if available
        if self.use_real_embeddings:
            try:
                self.openai_client = OpenAI()
                logger.info("OpenAI client initialized for embeddings")
            except:
                self.use_real_embeddings = False
                logger.warning("OpenAI client initialization failed")

        # Load embedding cache if exists
        self._load_embedding_cache()

        # Initialize or load both harmonization tiers
        self._initialize_harmonization_tiers()

        # Load original harmonization report if it exists
        self._load_original_report()

        self.debug_verbose = os.getenv("BB_DEBUG_VERBOSE", "0") == "1"
        self._dbg = (
            lambda m: print(f"[harmonizer][{datetime.now(timezone.utc).isoformat()}] {m}")) if self.debug_verbose else (
            lambda *_: None)

    
    def enable_batch_mode(self):
        """Enable batch processing mode for better performance."""
        self.batch_processing = True
        self.similarity_cache.clear()  # Clear cache for fresh batch
        
    def disable_batch_mode(self):
        """Disable batch mode and optionally clear caches."""
        self.batch_processing = False
        # Keep cache for future use unless it's too large
        if len(self.similarity_cache) > 10000:
            # Keep only most recent 5000 entries
            items = list(self.similarity_cache.items())
            self.similarity_cache = dict(items[-5000:])

    def _load_embedding_cache(self):
            """Load cached embeddings if available."""
            if self.embedding_cache_file.exists():
                try:
                    import pickle
                    with open(self.embedding_cache_file, 'rb') as f:
                        self.embedding_cache = pickle.load(f)
                    logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
                except Exception as e:
                    logger.warning(f"Could not load embedding cache: {e}")

    def _save_embedding_cache(self):
        """Save embeddings to cache."""
        try:
            import pickle
            self.embedding_cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.embedding_cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
        except Exception as e:
            logger.warning(f"Could not save embedding cache: {e}")

    def _initialize_harmonization_tiers(self):
        """Initialize or load both harmonization tiers."""
        if self.general_harmony_file.exists():
            self._load_harmonization_tier(self.general_harmony_file, self.general_groups, "general")
        else:
            logger.info("No general harmonization file found. Will build from data.")

        if self.specific_harmony_file.exists():
            self._load_harmonization_tier(self.specific_harmony_file, self.specific_groups, "specific")
        else:
            logger.info("No specific harmonization file found. Will build from data.")

    def _load_original_report(self):
        """Load the original harmonization report to get frequencies."""
        if self.harmonization_report_path.exists():
            logger.info(f"Loading base report from {self.harmonization_report_path}")

            with open(self.harmonization_report_path, 'r') as f:
                report = json.load(f)

            for category in ['topic', 'tone', 'intent']:
                if category in report and 'groups' in report[category]:
                    groups = report[category]['groups']

                    for group_name, group_members in groups.items():
                        frequency = len(group_members)
                        self.label_frequencies[category][group_name] = frequency

                        for member in group_members:
                            if member != group_name:
                                self.label_mappings[category][member] = group_name
                            if member not in self.label_frequencies[category]:
                                self.label_frequencies[category][member] = 1

                # Load explicit frequencies if available
                if category in report and 'frequencies' in report[category]:
                    for label, freq in report[category]['frequencies'].items():
                        self.label_frequencies[category][label] = max(
                            self.label_frequencies[category].get(label, 0),
                            freq
                        )

            # Build initial two-tier groups if they don't exist
            if not self.general_harmony_file.exists() or not self.specific_harmony_file.exists():
                logger.info("Building initial two-tier groups...")
                for category in ['topic', 'tone', 'intent']:
                    if self.label_frequencies[category]:
                        self._update_harmonization_tier(category)
                self.save_harmonization_tiers()

    def _load_harmonization_tier(self, file_path: Path, groups_dict: Dict, tier_name: str):
        """Load a harmonization tier from file with error handling."""
        logger.info(f"Loading {tier_name} harmonization from {file_path}")

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            for category in ['topic', 'tone', 'intent']:
                if category in data and 'groups' in data[category]:
                    groups_dict[category] = data[category]['groups']
                    logger.info(f"  {category}: Loaded {len(groups_dict[category])} {tier_name} groups")
        except json.JSONDecodeError as e:
            logger.error(f"  ❌ JSON decode error in {file_path}: {e}")
            logger.info(f"  Creating empty {tier_name} structure")
            # Initialize with empty structure
            for category in ['topic', 'tone', 'intent']:
                groups_dict[category] = {}
        except Exception as e:
            logger.error(f"  ❌ Error loading {file_path}: {e}")
            # Initialize with empty structure
            for category in ['topic', 'tone', 'intent']:
                groups_dict[category] = {}

    def get_embedding(self, text: str):
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text (cached or compute)."""
        if text in self.embedding_cache:
            self._dbg(f"embed cache HIT for '{text[:48]}'")
            # Debug: cached embedding found
            return self.embedding_cache[text]

        if self.use_real_embeddings:
            try:
                t0 = time.time()
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                embedding = np.array(response.data[0].embedding)
                self._dbg(f"embed API CALL for '{text[:48]}' took {time.time() - t0:.3f}s")
            except Exception as e:
                self._dbg(f"embed API FAILED for '{text[:48]}': {e}; falling back to simulated embedding")
                embedding = self._simulated_embedding(text)
        else:
            embedding = self._simulated_embedding(text)

        self.embedding_cache[text] = embedding
        return embedding

    def _get_embeddings_batch(self, texts: List[str], pre_filter: bool = True) -> Dict[str, np.ndarray]:
        """
        Get embeddings for multiple texts with smart filtering.
        Pre-filters using string similarity to reduce API calls.
        """
        results = {}
        uncached = []

        # First get all cached embeddings
        for text in texts:
            if text in self.embedding_cache:
                results[text] = self.embedding_cache[text]
            else:
                uncached.append(text)

        if not uncached:
            return results

        # Pre-filter if enabled to reduce API calls
        if pre_filter and len(uncached) > 100:
            # Group similar strings together
            groups = self._group_similar_strings(uncached)

            # Only get embeddings for group representatives
            representatives = [group[0] for group in groups.values()]
            logger.info(f"Pre-filtered {len(uncached)} texts to {len(representatives)} representatives")

            # Get embeddings for representatives
            if self.use_real_embeddings and representatives:
                batch_size = 2048
                for i in range(0, len(representatives), batch_size):
                    batch = representatives[i:i+batch_size]
                    try:
                        response = self.openai_client.embeddings.create(
                            model="text-embedding-3-small",
                            input=batch
                        )
                        for text, embedding_data in zip(batch, response.data):
                            embedding = np.array(embedding_data.embedding)
                            self.embedding_cache[text] = embedding
                            results[text] = embedding
                    except Exception as e:
                        logger.warning(f"Batch embedding failed: {e}")
                        for text in batch:
                            embedding = self._simulated_embedding(text)
                            self.embedding_cache[text] = embedding
                            results[text] = embedding

            # Apply representative embeddings to similar strings
            for rep, members in groups.items():
                if rep in results:
                    rep_embedding = results[rep]
                    for member in members[1:]:  # Skip the representative itself
                        # Add small noise to make embeddings slightly different
                        noise = np.random.normal(0, 0.01, rep_embedding.shape)
                        member_embedding = rep_embedding + noise
                        self.embedding_cache[member] = member_embedding
                        results[member] = member_embedding
        else:
            # Small batch - get embeddings directly
            if self.use_real_embeddings:
                try:
                    response = self.openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=uncached
                    )
                    for text, embedding_data in zip(uncached, response.data):
                        embedding = np.array(embedding_data.embedding)
                        self.embedding_cache[text] = embedding
                        results[text] = embedding
                except Exception as e:
                    logger.warning(f"Embedding failed: {e}")
                    for text in uncached:
                        embedding = self._simulated_embedding(text)
                        self.embedding_cache[text] = embedding
                        results[text] = embedding
            else:
                for text in uncached:
                    embedding = self._simulated_embedding(text)
                    self.embedding_cache[text] = embedding
                    results[text] = embedding

        # Save cache periodically
        if len(self.embedding_cache) % 1000 == 0:
            self._save_embedding_cache()

        return results

    def _group_similar_strings(self, texts: List[str], threshold: float = 0.85) -> Dict[str, List[str]]:
        """Group similar strings together to reduce embedding API calls."""
        groups = {}
        processed = set()

        for text in texts:
            if text in processed:
                continue

            # This text becomes a group representative
            group = [text]
            processed.add(text)

            # Find similar texts
            for other in texts:
                if other in processed:
                    continue

                # Use string similarity for pre-filtering
                similarity = self.compute_string_similarity(text, other)
                if similarity >= threshold:
                    group.append(other)
                    processed.add(other)
                    # Removed debug print for performance

            groups[text] = group

        return groups

    def _simulated_embedding(self, text: str) -> np.ndarray:
        """Create a simulated embedding based on text features."""
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

        # Pad to standard embedding size
        while len(features) < 1536:
            features.append(0)

        return np.array(features[:1536])

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def compute_string_similarity(self, label1: str, label2: str) -> float:
        """Compute similarity using string matching (no embeddings)."""
        # Normalize for comparison
        l1 = label1.lower().replace('-', ' ').replace('_', ' ')
        l2 = label2.lower().replace('-', ' ').replace('_', ' ')

        # Direct match
        if l1 == l2:
            return 1.0

        # Check for subset relationships
        if l1 in l2 or l2 in l1:
            return 0.9

        # Check for word overlap
        words1 = set(l1.split())
        words2 = set(l2.split())
        if words1 and words2:
            overlap = len(words1 & words2)
            total = len(words1 | words2)
            if total > 0:
                jaccard = overlap / total
                if jaccard > 0.5:
                    return 0.7 + (jaccard * 0.2)

        # Use sequence matcher for fuzzy matching
        return SequenceMatcher(None, l1, l2).ratio()

    def compute_similarity(self, label1: str, label2: str, use_embeddings: bool = True) -> float:
        """
        Compute similarity between two labels.
        First tries string matching, then uses embeddings if needed.
        """
        # Check cache first
        cache_key = tuple(sorted([label1, label2]))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Quick string similarity check first
        string_sim = self.compute_string_similarity(label1, label2)

        # If string similarity is high enough, don't bother with embeddings
        if string_sim >= 0.85:
            # Cache the result
            self.similarity_cache[cache_key] = string_sim
        return string_sim

        # If string similarity is very low, don't waste API calls
        if string_sim < 0.3:
            return string_sim

        # For medium similarity, use embeddings if available
        if use_embeddings and self.use_real_embeddings:
            emb1 = self._get_embedding(label1)
            emb2 = self._get_embedding(label2)
            embedding_sim = self._cosine_similarity(emb1, emb2)

            # Weighted average of string and embedding similarity
            # Cache the result
            result = (string_sim * 0.3) + (embedding_sim * 0.7)
            self.similarity_cache[cache_key] = result
            return result

        return string_sim

    def _build_groups(self, labels_with_freq: List[Tuple[str, float]], threshold: float) -> Dict[str, List[str]]:
        """Build groups of labels based on similarity threshold."""
        # Deduplicate labels first (keep highest frequency)
        unique_labels = {}
        for label, freq in labels_with_freq:
            if label not in unique_labels or freq > unique_labels[label]:
                unique_labels[label] = freq
        
        # Sort by frequency
        sorted_labels = sorted(unique_labels.items(), key=lambda x: x[1], reverse=True)

        groups = {}
        assigned = set()

        # Pre-filter candidates for efficiency
        label_texts = [label for label, _ in sorted_labels]

        # Get embeddings for all labels at once (with pre-filtering)
        if self.use_real_embeddings and len(label_texts) > 10:
            embeddings = self._get_embeddings_batch(label_texts, pre_filter=True)
        else:
            embeddings = {}

        for label, freq in sorted_labels:
            if label in assigned:
                continue

            # This label becomes a group canonical
            group = [label]
            assigned.add(label)

            # Find similar labels
            for other_label, other_freq in sorted_labels:
                if other_label in assigned:
                    continue

                # Compute similarity
                if embeddings and label in embeddings and other_label in embeddings:
                    # Use embedding similarity
                    similarity = self._cosine_similarity(embeddings[label], embeddings[other_label])
                    # Removed debug print for performance
                else:
                    # Fall back to string similarity
                    similarity = self.compute_string_similarity(label, other_label)
                    # Removed debug print for performance

                if similarity >= threshold:
                    group.append(other_label)
                    assigned.add(other_label)

            # Store group with canonical as key
            groups[label] = group

        return groups

    def _update_harmonization_tier(self, category: str):
        """Update both harmonization tiers for a category."""
        # Get all labels with frequencies
        labels_with_freq = list(self.label_frequencies[category].items())

        if not labels_with_freq:
            return

        # Build general groups (broader, threshold > 0.5)
        general = self._build_groups(labels_with_freq, self.general_threshold)
        self.general_groups[category] = general

        # Build specific groups (fine-grained, threshold > 0.8)
        specific = self._build_groups(labels_with_freq, self.specific_threshold)
        self.specific_groups[category] = specific

        logger.info(f"  {category}: {len(general)} general groups, {len(specific)} specific groups")

    def find_canonical(self, label: str, category: str) -> Tuple[str, str, str]:
        """
        Find canonical forms at both tiers.
        Returns (label, specific_canonical, general_canonical)
        """
        specific_canonical = label
        general_canonical = label

        # Check specific groups first (more restrictive)
        for canonical, members in self.specific_groups[category].items():
            if label in members:
                specific_canonical = canonical
                break

        # Check general groups (broader)
        for canonical, members in self.general_groups[category].items():
            if label in members:
                general_canonical = canonical
                break

        return label, specific_canonical, general_canonical

    def harmonize_label_set(self, labels: List[Dict], category: str) -> Tuple[List[Dict], Dict]:
        """
        Harmonize a set of labels using two-tier system.
        Returns (harmonized_labels, harmonization_info)
        """
        if not labels:
            return [], {}

        harmonized = []
        info = {
            'original_count': len(labels),
            'specific_groups_used': set(),
            'general_groups_used': set(),
            'similarity_index': 0.0
        }

        # Process each label
        for item in labels:
            label = item.get('label', '')
            prob = item.get('p', item.get('probability', 1.0))

            if not label:
                continue

            # Update frequency
            self.label_frequencies[category][label] += prob

            # Get both canonical forms
            _, specific_canonical, general_canonical = self.find_canonical(label, category)

            # Create harmonized entry with probability preserved
            harmonized.append({
                'label': specific_canonical,
                'p': prob,
                'original': label,
                'general_group': general_canonical,
                'specific_group': specific_canonical
            })

            info['specific_groups_used'].add(specific_canonical)
            info['general_groups_used'].add(general_canonical)

        # Calculate similarity index
        if len(info['general_groups_used']) > 0:
            info['similarity_index'] = len(info['specific_groups_used']) / len(info['general_groups_used'])

        # Convert sets to counts for JSON serialization
        info['specific_groups_count'] = len(info['specific_groups_used'])
        info['general_groups_count'] = len(info['general_groups_used'])
        del info['specific_groups_used']
        del info['general_groups_used']

        # Merge duplicates while preserving max probability
        merged = {}
        for item in harmonized:
            label = item['label']
            if label in merged:
                merged[label]['p'] = max(merged[label]['p'], item['p'])
            else:
                merged[label] = item

        # Sort by probability
        harmonized = sorted(merged.values(), key=lambda x: x['p'], reverse=True)

        info['harmonized_count'] = len(harmonized)

        # Update co-occurrence for context building
        for i, item1 in enumerate(harmonized):
            for item2 in harmonized[i+1:]:
                weight = item1['p'] * item2['p']
                self.co_occurrence[category][item1['label']][item2['label']] += weight
                self.co_occurrence[category][item2['label']][item1['label']] += weight

        return harmonized, info

    def get_label_similarity_scores(self, query_labels: List[Dict], candidate_labels: List[Dict], category: str) -> float:
        """
        Calculate similarity between two sets of labels using probabilities as weights.
        Used for finding similar messages for context.
        """
        if not query_labels or not candidate_labels:
            return 0.0

        total_similarity = 0.0
        total_weight = 0.0

        for q_item in query_labels:
            q_label = q_item.get('label', '')
            q_prob = q_item.get('p', 1.0)

            for c_item in candidate_labels:
                c_label = c_item.get('label', '')
                c_prob = c_item.get('p', 1.0)

                # Compute similarity (uses caching and smart filtering)
                similarity = self.compute_similarity(q_label, c_label, use_embeddings=True)

                # Weight by both probabilities
                weighted_sim = similarity * q_prob * c_prob
                total_similarity += weighted_sim
                total_weight += q_prob * c_prob

        if total_weight > 0:
            return total_similarity / total_weight

        return 0.0

    def save_harmonization_tiers(self):
        """Save both harmonization tiers to files."""
        # Save general harmonization
        general_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'threshold': self.general_threshold,
            'tier': 'general',
            'description': 'Broader groupings for general concepts'
        }

        for category in ['topic', 'tone', 'intent']:
            general_data[category] = {
                'groups': self.general_groups[category],
                'group_count': len(self.general_groups[category]),
                'total_labels': sum(len(members) for members in self.general_groups[category].values())
            }

        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.general_harmony_file, 'w') as f:
            json.dump(general_data, f, indent=2)

        # Save specific harmonization
        specific_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'threshold': self.specific_threshold,
            'tier': 'specific',
            'description': 'Fine-grained groupings for specific concepts'
        }

        for category in ['topic', 'tone', 'intent']:
            specific_data[category] = {
                'groups': self.specific_groups[category],
                'group_count': len(self.specific_groups[category]),
                'total_labels': sum(len(members) for members in self.specific_groups[category].values())
            }

        with open(self.specific_harmony_file, 'w') as f:
            json.dump(specific_data, f, indent=2)

        logger.info(f"Saved harmonization tiers to {self.data_dir}")

    def get_harmonization_report(self) -> Dict:
        """Generate comprehensive two-tier harmonization report."""
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'configuration': {
                'similarity_threshold': self.similarity_threshold,
                'min_semantic_distance': self.min_semantic_distance,
                'use_real_embeddings': self.use_real_embeddings,
                'embeddings_cached': len(self.embedding_cache)
            }
        }

        for category in ['topic', 'tone', 'intent']:
            general_count = len(self.general_groups[category])
            specific_count = len(self.specific_groups[category])
            total_labels = len(self.label_frequencies[category])

            report[category] = {
                'total_unique_labels': total_labels,
                'general_groups': general_count,
                'specific_groups': specific_count,
                'top_labels': self.label_frequencies[category].most_common(20)
            }

        return report

    def process_label_set(self, labels: Dict[str, List[Dict]], message_context: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Process a label set (for compatibility with existing interface).
        """
        harmonized = {}

        for category in ['topic', 'tone', 'intent']:
            if category in labels:
                canonical_labels, info = self.harmonize_label_set(labels[category], category)
                harmonized[category] = canonical_labels

        # Periodically save harmonization tiers
        if not hasattr(self, '_process_count'):
            self._process_count = 0
        self._process_count += 1

        if self._process_count % 100 == 0:
            self.save_harmonization_tiers()
            self._save_embedding_cache()

        return harmonized


class EnhancedSmartLabelImporter:
    """Import labeled messages with two-tier harmonization."""

    def __init__(
        self,
        graph_store: GraphStore,
        data_dir: str = "data",
        use_harmonization_report: bool = True
    ):
        self.graph = graph_store
        self.data_dir = Path(data_dir)
        self.labels_dir = self.data_dir / "labels"

        # Initialize enhanced harmonizer
        harmonization_report_path = self.data_dir / "harmonization_report.json"

        logger.info("Initializing two-tier harmonization system...")
        self.harmonizer = EnhancedLabelHarmonizer(
            harmonization_report_path=str(harmonization_report_path),
            similarity_threshold=0.8,  # Specific threshold
            min_semantic_distance=0.5,  # General threshold
            use_real_embeddings=bool(os.getenv('OPENAI_API_KEY'))
        )

        self.stats = defaultdict(int)
        self.gid_to_node_id = {}
        self.label_node_ids = {}

    def load_message_index(self) -> Dict[str, Dict]:
        """Load the message index from chunks or other source."""
        chunks_dir = self.data_dir / "chunks"
        messages = {}

        if chunks_dir.exists():
            logger.info(f"Loading messages from chunks directory: {chunks_dir}")

            chunk_files = list(chunks_dir.glob("*.ndjson")) + list(chunks_dir.glob("*.json"))

            for chunk_file in chunk_files[:100]:  # Limit for testing
                try:
                    if chunk_file.suffix == '.ndjson':
                        with open(chunk_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                msg = json.loads(line)
                                gid = msg.get('gid', msg.get('id', ''))
                                if gid:
                                    messages[gid] = msg
                    else:
                        with open(chunk_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                for msg in data:
                                    gid = msg.get('gid', msg.get('id', ''))
                                    if gid:
                                        messages[gid] = msg
                            elif isinstance(data, dict):
                                gid = data.get('gid', data.get('id', ''))
                                if gid:
                                    messages[gid] = data
                except Exception as e:
                    logger.debug(f"Error loading {chunk_file}: {e}")

        logger.info(f"  Loaded metadata for {len(messages)} messages")
        return messages

    def sanitize_tag(self, tag: str) -> str:
        """Convert a label to a valid tag format."""
        tag = tag.lower()
        tag = ''.join(c if c.isalnum() or c in '_-' else '_' for c in tag)
        tag = '_'.join(filter(None, tag.split('_')))
        return tag[:30]

    def import_labeled_message(self, label_file: Path, message_index: Dict) -> Optional[str]:
        """Import a single labeled message with two-tier harmonization."""
        # Load label data
        with open(label_file, 'r', encoding='utf-8') as f:
            label_data = json.load(f)

        gid = label_data.get('gid', '')
        if not gid:
            return None

        # Get message metadata
        msg_meta = message_index.get(gid, {})

        # Harmonize labels with two-tier system
        canonical_topics, topic_info = self.harmonizer.harmonize_label_set(
            label_data.get('topic', []), 'topic'
        )
        canonical_tones, tone_info = self.harmonizer.harmonize_label_set(
            label_data.get('tone', []), 'tone'
        )
        canonical_intents, intent_info = self.harmonizer.harmonize_label_set(
            label_data.get('intent', []), 'intent'
        )

        # Track statistics
        self.stats['total_similarity_index'] += (
            topic_info.get('similarity_index', 0) +
            tone_info.get('similarity_index', 0) +
            intent_info.get('similarity_index', 0)
        ) / 3

        # Create tags using both tiers
        tags = []
        if canonical_topics:
            # Add specific group tag
            if canonical_topics[0].get('specific_group'):
                tags.append('s_' + self.sanitize_tag(canonical_topics[0]['specific_group']))
            # Add general group tag
            if canonical_topics[0].get('general_group'):
                tags.append('g_' + self.sanitize_tag(canonical_topics[0]['general_group']))

        if canonical_tones:
            tags.append('tone_' + self.sanitize_tag(canonical_tones[0]['label']))

        # Add role as tag
        role = msg_meta.get('role', 'unknown')
        tags.append(role)

        # Create message node with harmonized labels including probabilities
        node_attrs = {
            'gid': gid,
            'conversation_id': msg_meta.get('conversation_id'),
            'seq': msg_meta.get('seq'),
            'role': role,
            'confidence': label_data.get('confidence', 0),
            'topics': canonical_topics,  # Includes p values
            'tones': canonical_tones,    # Includes p values
            'intents': canonical_intents, # Includes p values
            'two_tier_harmonization': {
                'topic': topic_info,
                'tone': tone_info,
                'intent': intent_info
            }
        }

        # Create the node
        node_id = self.graph.create_node(
            type="log",
            text=msg_meta.get('content_preview', ''),
            tags=tags,
            attrs=node_attrs
        )

        self.gid_to_node_id[gid] = node_id
        self.stats['messages_imported'] += 1
        self.stats[f'messages_{role}'] += 1

        # Create label nodes with tier information
        self._create_tiered_label_nodes(canonical_topics, 'topic', node_id)
        self._create_tiered_label_nodes(canonical_tones, 'tone', node_id)
        self._create_tiered_label_nodes(canonical_intents, 'intent', node_id)

        return node_id

    def _create_tiered_label_nodes(self, labels: List[Dict], category: str, msg_id: str):
        """Create nodes for labels with two-tier information and probabilities."""
        for item in labels:
            label = item['label']
            prob = item['p']  # Preserve probability
            general_group = item.get('general_group', label)
            specific_group = item.get('specific_group', label)

            label_key = f"{category}_{label}"

            if label_key not in self.label_node_ids:
                # Create label node with tier tags
                label_tags = [category, 'label']
                if general_group != label:
                    label_tags.append('has_general')
                if specific_group != label:
                    label_tags.append('has_specific')
                label_tags.append(self.sanitize_tag(label)[:15])

                label_node_id = self.graph.create_node(
                    type="concept",
                    text=label,
                    tags=label_tags,
                    attrs={
                        'frequency': self.harmonizer.label_frequencies[category].get(label, 0),
                        'category': category,
                        'general_group': general_group,
                        'specific_group': specific_group
                    }
                )
                self.label_node_ids[label_key] = label_node_id

            # Connect to message with probability as weight
            label_node_id = self.label_node_ids[label_key]
            edge_id = self.graph.create_edge(
                src_id=msg_id,
                dst_id=label_node_id,
                kind="relates_to",
                weight=prob,  # Use probability as edge weight
                attrs={
                    'category': category,
                    'probability': prob,
                    'general_group': general_group,
                    'specific_group': specific_group
                }
            )
            self.stats['edges_created'] += 1

    def import_all_labels(self, max_files: Optional[int] = None) -> Dict:
        """Import all label files with two-tier harmonization."""
        # Load message index
        message_index = self.load_message_index()

        # Get all label files
        label_files = list(self.labels_dir.glob("*.json"))
        if max_files:
            label_files = label_files[:max_files]

        total = len(label_files)
        logger.info(f"\n📊 Found {total} label files to import")

        # Process each label file
        for i, label_file in enumerate(label_files):
            if i % 100 == 0:
                logger.info(f"  Processing {i+1}/{total} ({i*100//total}%)")

                # Update groups periodically
                if i > 0 and i % 500 == 0:
                    logger.info("    Updating two-tier groups...")
                    for category in ['topic', 'tone', 'intent']:
                        self.harmonizer._update_harmonization_tier(category)

            try:
                self.import_labeled_message(label_file, message_index)
            except Exception as e:
                logger.debug(f"Error processing {label_file.name}: {e}")
                self.stats['errors'] += 1

        # Final update of two-tier groups
        logger.info("\n🔄 Finalizing two-tier groups...")
        for category in ['topic', 'tone', 'intent']:
            self.harmonizer._update_harmonization_tier(category)

        # Save everything
        logger.info("\n💾 Saving data...")
        self.graph.save()
        self.harmonizer.save_harmonization_tiers()
        self.harmonizer._save_embedding_cache()

        # Generate report
        harmony_report = self.harmonizer.get_harmonization_report()
        report_path = self.data_dir / "two_tier_harmonization_report.json"
        with open(report_path, 'w') as f:
            json.dump(harmony_report, f, indent=2)

        # Print results
        logger.info("\n✅ Import complete!")
        logger.info(f"\n📈 Statistics:")
        logger.info(f"  Messages imported: {self.stats['messages_imported']}")
        logger.info(f"  Edges created: {self.stats['edges_created']}")
        logger.info(f"  Embeddings cached: {len(self.harmonizer.embedding_cache)}")

        return {
            'stats': dict(self.stats),
            'harmony_report': harmony_report
        }


def main():
    """Run the enhanced smart label import."""
    import sys

    # Add BiggerBrother to path if needed
    sys.path.insert(0, r'C:\BiggerBrother')

    # Initialize graph store
    graph = GraphStore(data_dir="C:/BiggerBrother/data/graph")

    # Create enhanced importer
    importer = EnhancedSmartLabelImporter(
        graph,
        data_dir="C:/BiggerBrother-minimal/data",
        use_harmonization_report=True
    )

    # Import all labels
    print("\nStarting enhanced label import with smart embedding usage...")
    print("This system:")
    print("  - Uses string matching first to filter candidates")
    print("  - Only calls embedding API for ambiguous cases")
    print("  - Groups similar strings to minimize API calls")
    print("  - Preserves label probabilities for context weighting")
    print("\nPress Ctrl+C to stop at any time\n")

    try:
        results = importer.import_all_labels(max_files=None)
    except KeyboardInterrupt:
        print("\n\nImport interrupted by user")
        print("Saving current state...")
        graph.save()
        importer.harmonizer.save_harmonization_tiers()
        importer.harmonizer._save_embedding_cache()

    print("\n✅ Enhanced import complete!")


if __name__ == "__main__":
    main()