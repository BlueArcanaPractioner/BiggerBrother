"""
SEMANTIC HARMONIZER V2 - Label Topology Manager
Handles label normalization, similarity grouping, and semantic drift tracking
"""

import json
import pickle
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class LabelNode:
    """Represents a label in the semantic space."""
    label: str
    canonical_form: str
    embedding: Optional[np.ndarray] = None
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    occurrence_count: int = 1
    merge_group: Optional[str] = None
    parent_group: Optional[str] = None
    confidence_history: List[float] = field(default_factory=list)
    context_examples: List[str] = field(default_factory=list)  # Example messages
    behavioral_correlation: float = 0.0  # Success rate when this label appears


@dataclass 
class SemanticGroup:
    """Represents a group of related labels."""
    group_id: str
    canonical_label: str
    members: Set[str] = field(default_factory=set)
    centroid_embedding: Optional[np.ndarray] = None
    stability_score: float = 1.0  # How stable this grouping has been
    behavioral_outcomes: Dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    

class EnhancedHarmonizer:
    """
    Advanced harmonizer that manages semantic label topology for the context engine.
    Handles normalization, grouping, drift detection, and behavioral correlation.
    """

    def __init__(
            self,
            data_dir: str = "data",
            strict_threshold: float = 0.85,
            loose_threshold: float = 0.65,
            drift_threshold: float = 0.15,
            cache_file: Optional[str] = None
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.strict_threshold = strict_threshold
        self.loose_threshold = loose_threshold
        self.drift_threshold = drift_threshold

        # FIX: Ensure cache_file is always a Path object
        if cache_file:
            self.cache_file = Path(cache_file)
        else:
            self.cache_file = self.data_dir / "harmonizer_cache.pkl"

        # Core data structures
        self.label_nodes: Dict[str, LabelNode] = {}
        self.semantic_groups: Dict[str, SemanticGroup] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}

        # Tracking structures
        self.drift_log: List[Dict] = []
        self.novel_labels: List[Tuple[str, datetime]] = []

        # Category-specific groups
        self.category_groups: Dict[str, Dict[str, SemanticGroup]] = {
            'topic': {},
            'tone': {},
            'intent': {}
        }

        # Load existing state
        self.load_state()
    
    def process_label_set(
        self, 
        labels: Dict[str, List[Dict[str, Any]]],
        message_context: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a complete label set from the labeler.
        
        Args:
            labels: Dict with 'topic', 'tone', 'intent' keys, each containing
                   list of {'label': str, 'p': float} dicts
            message_context: Optional message text for context
            
        Returns:
            Harmonized label set with canonical forms and groups
        """
        harmonized = {}
        
        for category, label_list in labels.items():
            harmonized[category] = []
            
            for item in label_list:
                label = item['label']
                confidence = item['p']
                
                # Get or create label node
                node = self.get_or_create_label_node(
                    label, category, confidence, message_context
                )
                
                # Find or assign group
                group = self.assign_to_group(node, category)
                
                harmonized[category].append({
                    'original': label,
                    'canonical': node.canonical_form,
                    'group': group.group_id,
                    'p': confidence,
                    'is_novel': node.occurrence_count == 1,
                    'drift_detected': self.check_drift(node, group)
                })
        
        # Detect cross-category patterns
        self.detect_label_correlations(harmonized)
        
        return harmonized
    
    def get_or_create_label_node(
        self,
        label: str,
        category: str,
        confidence: float,
        context: Optional[str] = None
    ) -> LabelNode:
        """Get existing or create new label node."""
        normalized = self.normalize_label(label)
        key = f"{category}:{normalized}"
        
        if key in self.label_nodes:
            node = self.label_nodes[key]
            node.last_seen = datetime.now(timezone.utc)
            node.occurrence_count += 1
            node.confidence_history.append(confidence)
            if context and len(node.context_examples) < 5:
                node.context_examples.append(context[:100])
        else:
            # Create new node
            node = LabelNode(
                label=label,
                canonical_form=normalized,
                confidence_history=[confidence]
            )
            if context:
                node.context_examples.append(context[:100])
            
            self.label_nodes[key] = node
            self.novel_labels.append((key, node.first_seen))
            
            # Get embedding for new label
            node.embedding = self.get_embedding(normalized)
        
        return node
    
    def assign_to_group(self, node: LabelNode, category: str) -> SemanticGroup:
        """
        Assign a label node to a semantic group using similarity matching.
        Creates new group if no good match found.
        """
        if node.merge_group:
            # Already assigned
            return self.semantic_groups[node.merge_group]
        
        if node.embedding is None:
            node.embedding = self.get_embedding(node.canonical_form)
        
        category_groups = self.category_groups[category]
        
        # Find best matching group using embeddings
        best_group = None
        best_similarity = 0.0
        
        for group in category_groups.values():
            if group.centroid_embedding is not None:
                similarity = self.cosine_similarity(
                    node.embedding,
                    group.centroid_embedding
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_group = group
        
        # Decide whether to join existing group or create new one
        if best_similarity >= self.strict_threshold:
            # Join existing tight group
            best_group.members.add(node.canonical_form)
            node.merge_group = best_group.group_id
            self.update_group_centroid(best_group)
            
        elif best_similarity >= self.loose_threshold:
            # Create subgroup under parent
            parent_group = best_group
            new_group = self.create_group(
                node.canonical_form,
                category,
                parent=parent_group.group_id
            )
            node.merge_group = new_group.group_id
            node.parent_group = parent_group.group_id
            best_group = new_group
            
        else:
            # Create entirely new group
            best_group = self.create_group(node.canonical_form, category)
            node.merge_group = best_group.group_id
        
        return best_group
    
    def create_group(
        self,
        canonical_label: str,
        category: str,
        parent: Optional[str] = None
    ) -> SemanticGroup:
        """Create a new semantic group."""
        group_id = f"{category}:{canonical_label}:{datetime.now(timezone.utc).timestamp()}"
        
        group = SemanticGroup(
            group_id=group_id,
            canonical_label=canonical_label,
            members={canonical_label}
        )
        
        # Set centroid as label embedding initially
        group.centroid_embedding = self.get_embedding(canonical_label)
        
        self.semantic_groups[group_id] = group
        self.category_groups[category][group_id] = group
        
        return group
    
    def update_group_centroid(self, group: SemanticGroup):
        """Update the centroid embedding for a group based on its members."""
        if not group.members:
            return
        
        embeddings = []
        for member in group.members:
            # Find the label node
            for key, node in self.label_nodes.items():
                if node.canonical_form == member:
                    if node.embedding is not None:
                        embeddings.append(node.embedding)
                    break
        
        if embeddings:
            # Compute mean embedding as centroid
            group.centroid_embedding = np.mean(embeddings, axis=0)
            group.last_updated = datetime.now(timezone.utc)
    
    def check_drift(self, node: LabelNode, group: SemanticGroup) -> bool:
        """
        Check if a label is drifting from its assigned group.
        """
        if len(node.confidence_history) < 10:
            return False
        
        # Check if recent confidences differ from historical
        recent_conf = np.mean(node.confidence_history[-5:])
        historical_conf = np.mean(node.confidence_history[:-5])
        
        drift = abs(recent_conf - historical_conf)
        
        if drift > self.drift_threshold:
            self.drift_log.append({
                'label': node.canonical_form,
                'group': group.group_id,
                'drift_amount': drift,
                'timestamp': datetime.now(timezone.utc)
            })
            return True
        
        return False
    
    def detect_label_correlations(self, labels: Dict):
        """
        Detect correlations between labels across categories.
        Useful for finding behavioral patterns.
        """
        # This is where we'd track co-occurrence patterns
        # For now, just count co-occurrences
        pass
    
    def get_similar_labels(
        self,
        label: str,
        category: str,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find k most similar labels to the given one.
        """
        target_embedding = self.get_embedding(self.normalize_label(label))
        
        similarities = []
        for key, node in self.label_nodes.items():
            if key.startswith(f"{category}:") and node.embedding is not None:
                sim = self.cosine_similarity(target_embedding, node.embedding)
                similarities.append((node.canonical_form, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def update_behavioral_correlation(
        self,
        labels: Dict[str, List[str]],
        behavior_occurred: bool
    ):
        """
        Update behavioral correlation scores for labels based on outcomes.
        """
        for category, label_list in labels.items():
            for label in label_list:
                key = f"{category}:{self.normalize_label(label)}"
                if key in self.label_nodes:
                    node = self.label_nodes[key]
                    # Simple exponential moving average
                    alpha = 0.1
                    node.behavioral_correlation = (
                        alpha * (1.0 if behavior_occurred else 0.0) +
                        (1 - alpha) * node.behavioral_correlation
                    )
    
    def get_group_behavioral_stats(self, group_id: str) -> Dict:
        """
        Get behavioral statistics for a semantic group.
        """
        if group_id not in self.semantic_groups:
            return {}
        
        group = self.semantic_groups[group_id]
        
        # Aggregate stats from member nodes
        correlations = []
        for member in group.members:
            for key, node in self.label_nodes.items():
                if node.canonical_form == member:
                    correlations.append(node.behavioral_correlation)
        
        return {
            'group_id': group_id,
            'canonical': group.canonical_label,
            'member_count': len(group.members),
            'avg_behavioral_correlation': np.mean(correlations) if correlations else 0.0,
            'stability': group.stability_score
        }
    
    def normalize_label(self, label: str) -> str:
        """Normalize label for comparison."""
        # Remove extra spaces, lowercase, remove special chars
        normalized = label.lower().strip()
        normalized = ''.join(c for c in normalized if c.isalnum() or c in [' ', '_', '-'])
        normalized = ' '.join(normalized.split())  # Collapse multiple spaces
        return normalized
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text. Uses cache to avoid recomputation.
        In production, this would call the actual embedding API.
        """
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Placeholder: In production, call OpenAI embeddings API
        # For now, return random embedding for testing
        embedding = np.random.randn(1536)  # OpenAI embedding dimension
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        self.embedding_cache[text] = embedding
        return embedding
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def save_state(self):
        """Save harmonizer state to disk."""
        # Ensure cache_file is Path for consistency
        cache_path = Path(self.cache_file) if isinstance(self.cache_file, str) else self.cache_file

        state = {
            'label_nodes': self.label_nodes,
            'semantic_groups': self.semantic_groups,
            'embedding_cache': self.embedding_cache,
            'drift_log': self.drift_log,
            'novel_labels': self.novel_labels,
            'category_groups': self.category_groups
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self):
        """Load harmonizer state from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    state = pickle.load(f)
                    self.label_nodes = state.get('label_nodes', {})
                    self.semantic_groups = state.get('semantic_groups', {})
                    self.embedding_cache = state.get('embedding_cache', {})
                    self.drift_log = state.get('drift_log', [])
                    self.novel_labels = state.get('novel_labels', [])
                    self.category_groups = state.get('category_groups', {
                        'topic': {}, 'tone': {}, 'intent': {}
                    })
            except Exception as e:
                print(f"Could not load harmonizer state: {e}")
    
    def get_visual_topology(self, category: str) -> Dict:
        """
        Generate topology data for visualization.
        Returns nodes and edges suitable for force-directed graph.
        """
        nodes = []
        edges = []
        
        # Add group nodes
        for group in self.category_groups[category].values():
            nodes.append({
                'id': group.group_id,
                'label': group.canonical_label,
                'type': 'group',
                'size': len(group.members),
                'behavioral_score': self.get_group_behavioral_stats(group.group_id)[
                    'avg_behavioral_correlation'
                ]
            })
        
        # Add label nodes and edges to groups
        for key, node in self.label_nodes.items():
            if key.startswith(f"{category}:"):
                nodes.append({
                    'id': key,
                    'label': node.label,
                    'type': 'label',
                    'occurrences': node.occurrence_count,
                    'behavioral_correlation': node.behavioral_correlation
                })
                
                if node.merge_group:
                    edges.append({
                        'source': key,
                        'target': node.merge_group,
                        'weight': np.mean(node.confidence_history[-10:])
                            if node.confidence_history else 0.5
                    })
        
        return {'nodes': nodes, 'edges': edges}
