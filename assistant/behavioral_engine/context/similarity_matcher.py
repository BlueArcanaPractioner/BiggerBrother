"""
Context Similarity Matcher for BiggerBrother
============================================

Implements weighted similarity matching between messages using harmonized labels,
tier weights, and recency bias to build optimal context headers.
"""

from __future__ import annotations
import json
import os
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np

from app.openai_client import OpenAIClient
from app.label_integration_wrappers import LabelGenerator


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
        # Context size parameters
        context_minimum_char_long_term: int = 50000,
        context_minimum_char_recent: int = 10000,
        max_context_messages: int = 50,
        # Weighting parameters
        general_tier_weight: float = 0.3,
        specific_tier_weight: float = 0.7,
        recency_decay_factor: float = 0.95,  # Per day decay
        recency_cutoff_days: int = 30
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
        cache_file = self.harmonization_dir / "embedding_cache.pkl"
        if cache_file.exists():
            import pickle
            try:
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
            except:
                self.embedding_cache = {}
    
    def _save_embedding_cache(self):
        """Save embedding cache."""
        cache_file = self.harmonization_dir / "embedding_cache.pkl"
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(self.embedding_cache, f)
    
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
        threshold = 0.5 if tier == "general" else 0.8
        groups = self.general_groups if tier == "general" else self.specific_groups
        category_groups = groups.get(category, {}).get("groups", {})
        
        if not category_groups:
            return None
        
        # Get embedding for the label
        label_embedding = self._get_embedding(label)
        if label_embedding is None:
            return None
        
        # Sort groups by frequency (size) for checking highest frequency first
        sorted_groups = sorted(
            category_groups.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        best_similarity = 0
        best_group = None
        
        for canonical, variants in sorted_groups:
            # Get embedding for canonical label
            canonical_embedding = self._get_embedding(canonical)
            if canonical_embedding is None:
                continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(label_embedding, canonical_embedding)
            
            if similarity >= threshold and similarity > best_similarity:
                best_similarity = similarity
                best_group = canonical
        
        return best_group
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text, using cache when possible."""
        if text in self.embedding_cache:
            return np.array(self.embedding_cache[text])
        
        try:
            # This would normally call OpenAI embeddings API
            # For now, returning None to indicate we need to implement this
            # In production, you'd call: openai.Embedding.create(input=text, model="text-embedding-ada-002")
            return None
        except Exception as e:
            print(f"Error getting embedding for '{text}': {e}")
            return None
    
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
    
    def find_similar_messages(
        self,
        message: str,
        min_chars_recent: Optional[int] = None,
        min_chars_long_term: Optional[int] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Find similar messages and build context.
        
        Args:
            message: Current message to find context for
            min_chars_recent: Override for recent context minimum
            min_chars_long_term: Override for long-term context minimum
            
        Returns:
            Tuple of (context_messages, metadata)
        """
        min_chars_recent = min_chars_recent or self.context_minimum_char_recent
        min_chars_long_term = min_chars_long_term or self.context_minimum_char_long_term
        
        # Get and harmonize labels for current message
        current_labels = self.get_message_labels(message)
        current_harmonized = self.harmonize_and_score_labels(current_labels)
        
        # Create new chunk and label files for the message
        chunk_file, label_file = self._create_message_files(message, current_labels)
        
        # Find all existing labels and calculate similarities
        similarities = []
        
        for label_file_path in self.labels_dir.glob("*.json"):
            try:
                with open(label_file_path, 'r') as f:
                    target_labels = json.load(f)
                
                # Skip if no gid
                gid = target_labels.get("gid")
                if not gid:
                    continue
                
                # Get timestamp from filename or label
                timestamp_str = label_file_path.stem.split("_")[0]
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d")
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                except:
                    timestamp = datetime.now(timezone.utc) - timedelta(days=7)
                
                # Harmonize target labels
                target_harmonized = self.harmonize_and_score_labels(target_labels)
                
                # Calculate similarity
                similarity = self.calculate_similarity_score(
                    current_harmonized,
                    target_harmonized,
                    timestamp
                )
                
                if similarity > 0:
                    similarities.append({
                        "gid": gid,
                        "similarity": similarity,
                        "timestamp": timestamp,
                        "label_file": label_file_path.name
                    })
            
            except Exception as e:
                print(f"Error processing {label_file_path}: {e}")
                continue
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Build context by adding messages until we hit limits
        context_messages = []
        recent_chars = 0
        long_term_chars = 0
        
        for item in similarities[:self.max_context_messages]:
            # Load the chunk for this gid
            chunk_content = self._load_chunk_by_gid(item["gid"])
            if not chunk_content:
                continue
            
            content_length = len(chunk_content.get("content_text", ""))
            
            # Determine if this is recent (within 7 days)
            is_recent = (datetime.now(timezone.utc) - item["timestamp"]).days <= 7
            
            if is_recent and recent_chars < min_chars_recent:
                context_messages.append({
                    "gid": item["gid"],
                    "content": chunk_content.get("content_text", ""),
                    "similarity": item["similarity"],
                    "timestamp": item["timestamp"].isoformat(),
                    "is_recent": True
                })
                recent_chars += content_length
            
            elif long_term_chars < min_chars_long_term:
                context_messages.append({
                    "gid": item["gid"],
                    "content": chunk_content.get("content_text", ""),
                    "similarity": item["similarity"],
                    "timestamp": item["timestamp"].isoformat(),
                    "is_recent": False
                })
                long_term_chars += content_length
            
            # Stop if we've hit both limits
            if recent_chars >= min_chars_recent and long_term_chars >= min_chars_long_term:
                break
        
        metadata = {
            "current_labels": current_labels,
            "current_harmonized": current_harmonized,
            "total_similarities_found": len(similarities),
            "context_messages_included": len(context_messages),
            "recent_chars": recent_chars,
            "long_term_chars": long_term_chars,
            "chunk_file": chunk_file,
            "label_file": label_file
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