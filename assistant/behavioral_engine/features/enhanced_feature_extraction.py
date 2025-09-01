"""
Enhanced Feature Extraction System
===================================

Extracts behavioral features from conversations and activity logs
for pattern analysis and prediction.
"""

from __future__ import annotations
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
import statistics
import math

from assistant.logger.unified import UnifiedLogger
from assistant.graph.store import GraphStore

def ensure_timezone_aware(dt: Optional[datetime] = None) -> datetime:
    """Ensure datetime is timezone-aware. Use UTC if no timezone specified.
    
    Args:
        dt: A datetime object (naive or aware) or None
        
    Returns:
        A timezone-aware datetime object (in UTC)
    """
    if dt is None:
        return datetime.now(timezone.utc)
    if not hasattr(dt, 'tzinfo') or dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt



@dataclass
class MessageFeatures:
    """Features extracted from a single message."""
    
    # Basic metrics
    message_length: int
    word_count: int
    sentence_count: int
    
    # Vocabulary metrics
    unique_words: int
    vocabulary_richness: float  # unique_words / word_count
    rare_word_count: int  # Words outside top 1000
    rare_word_ratio: float
    
    # Linguistic features
    avg_word_length: float
    punctuation_count: int
    capital_ratio: float
    question_marks: int
    exclamation_marks: int
    
    # Temporal features
    hour_of_day: int
    day_of_week: int
    time_since_last_message: Optional[float] = None  # seconds
    
    # Context features
    messages_in_last_hour: int = 0
    messages_in_last_day: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return asdict(self)


@dataclass
class ActivityFeatures:
    """Features from activity logs in a time window."""
    
    # Activity counts
    total_logs: int
    unique_categories: int
    
    # Category breakdown
    category_counts: Dict[str, int] = field(default_factory=dict)
    
    # Productivity metrics
    tasks_completed: int = 0
    pomodoros_completed: int = 0
    focus_sessions: int = 0
    avg_focus_score: float = 0.0
    
    # Wellness metrics
    exercise_minutes: float = 0.0
    sleep_hours: float = 0.0
    mood_logs: int = 0
    avg_mood: float = 0.0
    avg_energy: float = 0.0
    
    # Study/Learning metrics
    study_sessions: int = 0
    study_hours: float = 0.0
    pages_read: int = 0
    
    # Temporal patterns
    most_active_hour: Optional[int] = None
    activity_spread: float = 0.0  # Standard deviation of activity times
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return asdict(self)


@dataclass
class SessionFeatures:
    """Aggregated features for a conversation session."""
    
    session_id: str
    start_time: datetime
    end_time: datetime
    duration_minutes: float
    
    # Message statistics
    total_messages: int
    user_messages: int
    assistant_messages: int
    avg_message_length: float
    avg_response_time: float  # seconds between messages
    
    # Vocabulary analysis
    total_words: int
    unique_words: int
    vocabulary_diversity: float
    rare_word_percentage: float
    
    # Topic/Label features
    dominant_labels: List[str] = field(default_factory=list)
    label_diversity: float = 0.0
    
    # Activity features during session
    activity_features: Optional[ActivityFeatures] = None
    
    # Behavioral indicators
    engagement_score: float = 0.0  # Based on response time and length
    productivity_score: float = 0.0  # Based on task/focus logs
    wellness_score: float = 0.0  # Based on mood/energy/exercise
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["start_time"] = self.start_time.isoformat()
        data["end_time"] = self.end_time.isoformat()
        return data


class EnhancedFeatureExtractor:
    """
    Extracts comprehensive features from messages and activity logs
    for behavioral analysis.
    """
    
    def __init__(
        self,
        logger: UnifiedLogger,
        graph_store: GraphStore,
        data_dir: str = "data/features"
    ):
        """Initialize feature extractor."""
        self.logger = logger
        self.graph = graph_store
        self.data_dir = data_dir
        
        os.makedirs(data_dir, exist_ok=True)
        
        # Load or build vocabulary statistics
        self.user_vocabulary = self._load_user_vocabulary()
        self.top_1000_words = self._load_top_words()
        
        # Cache for efficiency
        self.feature_cache = {}
        self.activity_cache = {}
    
    def _load_user_vocabulary(self) -> Counter:
        """Load user's vocabulary statistics."""
        vocab_file = os.path.join(self.data_dir, "user_vocabulary.json")
        if os.path.exists(vocab_file):
            with open(vocab_file, "r") as f:
                data = json.load(f)
                return Counter(data)
        return Counter()
    
    def _load_top_words(self) -> Set[str]:
        """Load top 1000 most common words for the user."""
        if self.user_vocabulary:
            return set(word for word, _ in self.user_vocabulary.most_common(1000))
        
        # Fallback to common English words
        common_words = set([
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their",
            "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
            "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
            "take", "people", "into", "year", "your", "good", "some", "could", "them",
            "see", "other", "than", "then", "now", "look", "only", "come", "its", "over",
            "think", "also", "back", "after", "use", "two", "how", "our", "work", "first",
            "well", "way", "even", "new", "want", "because", "any", "these", "give", "day",
            "most", "us", "is", "was", "are", "been", "has", "had", "were", "said", "did",
            "getting", "made", "find", "where", "much", "too", "very", "still", "being", "going"
        ])
        
        return common_words
    
    def extract_message_features(
        self,
        message: str,
        timestamp: datetime,
        previous_timestamp: Optional[datetime] = None,
        context_window_hours: int = 24
    ) -> MessageFeatures:
        """
        Extract features from a single message.
        
        Args:
            message: Message text
            timestamp: Message timestamp
            previous_timestamp: Timestamp of previous message
            context_window_hours: Hours to look back for context
            
        Returns:
            Extracted features
        """
        # Basic text metrics
        message_length = len(message)
        words = self._tokenize(message.lower())
        word_count = len(words)
        sentences = self._split_sentences(message)
        sentence_count = len(sentences)
        
        # Vocabulary metrics
        unique_words = len(set(words))
        vocabulary_richness = unique_words / word_count if word_count > 0 else 0
        
        # Rare words (outside top 1000)
        rare_words = [w for w in words if w not in self.top_1000_words]
        rare_word_count = len(rare_words)
        rare_word_ratio = rare_word_count / word_count if word_count > 0 else 0
        
        # Update user vocabulary
        self.user_vocabulary.update(words)
        
        # Linguistic features
        avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
        punctuation_count = len(re.findall(r'[.,;:!?]', message))
        capital_letters = sum(1 for c in message if c.isupper())
        capital_ratio = capital_letters / message_length if message_length > 0 else 0
        question_marks = message.count('?')
        exclamation_marks = message.count('!')
        
        # Temporal features
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()
        
        time_since_last = None
        if previous_timestamp:
            time_since_last = (timestamp - previous_timestamp).total_seconds()
        
        # Context features
        messages_in_last_hour = self._count_messages_in_window(
            timestamp - timedelta(hours=1),
            timestamp
        )
        messages_in_last_day = self._count_messages_in_window(
            timestamp - timedelta(hours=context_window_hours),
            timestamp
        )
        
        return MessageFeatures(
            message_length=message_length,
            word_count=word_count,
            sentence_count=sentence_count,
            unique_words=unique_words,
            vocabulary_richness=vocabulary_richness,
            rare_word_count=rare_word_count,
            rare_word_ratio=rare_word_ratio,
            avg_word_length=avg_word_length,
            punctuation_count=punctuation_count,
            capital_ratio=capital_ratio,
            question_marks=question_marks,
            exclamation_marks=exclamation_marks,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            time_since_last_message=time_since_last,
            messages_in_last_hour=messages_in_last_hour,
            messages_in_last_day=messages_in_last_day
        )
    
    def extract_activity_features(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> ActivityFeatures:
        """
        Extract features from activity logs in a time window.
        
        Args:
            start_time: Window start
            end_time: Window end
            
        Returns:
            Activity features
        """
        # Check cache
        cache_key = f"{start_time.isoformat()}_{end_time.isoformat()}"
        if cache_key in self.activity_cache:
            return self.activity_cache[cache_key]
        
        # Get logs in time window
        logs = self._get_logs_in_window(start_time, end_time)
        
        if not logs:
            features = ActivityFeatures(
                total_logs=0,
                unique_categories=0
            )
            self.activity_cache[cache_key] = features
            return features
        
        # Count categories
        category_counts = Counter()
        
        # Initialize metrics
        tasks_completed = 0
        pomodoros_completed = 0
        focus_sessions = 0
        focus_scores = []
        
        exercise_minutes = 0.0
        sleep_hours = 0.0
        mood_values = []
        energy_values = []
        
        study_sessions = 0
        study_hours = 0.0
        pages_read = 0
        
        activity_times = []
        
        for log in logs:
            category = log.get("category", "unknown")
            category_counts[category] += 1
            
            timestamp = ensure_timezone_aware(datetime.fromisoformat(log["timestamp"]))
            activity_times.append(timestamp.hour + timestamp.minute / 60)
            
            # Process by category
            if category == "task":
                if "completed" in str(log.get("value", "")).lower():
                    tasks_completed += 1
                if "pomodoro" in str(log.get("metadata", {})).lower():
                    pomodoros_completed += 1
            
            elif category == "focus":
                focus_sessions += 1
                if isinstance(log.get("value"), (int, float)):
                    focus_scores.append(float(log["value"]))
            
            elif category == "exercise":
                if isinstance(log.get("value"), (int, float)):
                    exercise_minutes += float(log["value"])
                elif "duration" in log.get("metadata", {}):
                    exercise_minutes += log["metadata"]["duration"]
            
            elif category == "sleep":
                if isinstance(log.get("value"), (int, float)):
                    sleep_hours += float(log["value"])
            
            elif category == "mood":
                if isinstance(log.get("value"), (int, float)):
                    mood_values.append(float(log["value"]))
            
            elif category == "energy":
                if isinstance(log.get("value"), (int, float)):
                    energy_values.append(float(log["value"]))
            
            elif category in ["study", "learning", "reading"]:
                study_sessions += 1
                if "duration" in log.get("metadata", {}):
                    study_hours += log["metadata"]["duration"] / 60
                if "pages" in log.get("metadata", {}):
                    pages_read += log["metadata"]["pages"]
        
        # Calculate aggregates
        avg_focus_score = statistics.mean(focus_scores) if focus_scores else 0.0
        avg_mood = statistics.mean(mood_values) if mood_values else 0.0
        avg_energy = statistics.mean(energy_values) if energy_values else 0.0
        
        # Temporal patterns
        most_active_hour = None
        activity_spread = 0.0
        
        if activity_times:
            hour_counts = Counter(int(t) for t in activity_times)
            most_active_hour = hour_counts.most_common(1)[0][0]
            activity_spread = statistics.stdev(activity_times) if len(activity_times) > 1 else 0.0
        
        features = ActivityFeatures(
            total_logs=len(logs),
            unique_categories=len(category_counts),
            category_counts=dict(category_counts),
            tasks_completed=tasks_completed,
            pomodoros_completed=pomodoros_completed,
            focus_sessions=focus_sessions,
            avg_focus_score=avg_focus_score,
            exercise_minutes=exercise_minutes,
            sleep_hours=sleep_hours,
            mood_logs=len(mood_values),
            avg_mood=avg_mood,
            avg_energy=avg_energy,
            study_sessions=study_sessions,
            study_hours=study_hours,
            pages_read=pages_read,
            most_active_hour=most_active_hour,
            activity_spread=activity_spread
        )
        
        # Cache results
        self.activity_cache[cache_key] = features
        
        return features
    
    def extract_session_features(
        self,
        session_id: str,
        messages: List[Dict],
        labels: List[List[str]]
    ) -> SessionFeatures:
        """
        Extract features for an entire conversation session.
        
        Args:
            session_id: Session identifier
            messages: List of message dictionaries
            labels: List of labels for each message
            
        Returns:
            Session features
        """
        if not messages:
            raise ValueError("No messages to analyze")
        
        # Sort messages by timestamp
        messages = sorted(messages, key=lambda m: m.get("timestamp", ""))
        
        # Parse timestamps
        timestamps = []
        for msg in messages:
            if isinstance(msg.get("timestamp"), str):
                timestamps.append(ensure_timezone_aware(datetime.fromisoformat(msg["timestamp"])))
            else:
                timestamps.append(datetime.now(timezone.utc))
        
        start_time = timestamps[0]
        end_time = timestamps[-1]
        duration_minutes = (end_time - start_time).total_seconds() / 60
        
        # Separate user and assistant messages
        user_messages = [m for m in messages if m.get("role") == "user"]
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        
        # Extract message features
        all_words = []
        message_lengths = []
        response_times = []
        
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            words = self._tokenize(content.lower())
            all_words.extend(words)
            message_lengths.append(len(content))
            
            if i > 0:
                response_time = (timestamps[i] - timestamps[i-1]).total_seconds()
                response_times.append(response_time)
        
        # Vocabulary analysis
        total_words = len(all_words)
        unique_words = len(set(all_words))
        vocabulary_diversity = unique_words / total_words if total_words > 0 else 0
        
        # Rare words
        rare_words = [w for w in all_words if w not in self.top_1000_words]
        rare_word_percentage = (len(rare_words) / total_words * 100) if total_words > 0 else 0
        
        # Label analysis
        all_labels = []
        for label_list in labels:
            if isinstance(label_list, list):
                all_labels.extend(label_list)
            else:
                all_labels.append(str(label_list))
        
        label_counts = Counter(all_labels)
        dominant_labels = [label for label, _ in label_counts.most_common(5)]
        label_diversity = len(set(all_labels)) / len(all_labels) if all_labels else 0
        
        # Get activity features for session window
        activity_features = self.extract_activity_features(start_time, end_time)
        
        # Calculate behavioral scores
        engagement_score = self._calculate_engagement_score(
            response_times,
            message_lengths
        )
        
        productivity_score = self._calculate_productivity_score(activity_features)
        wellness_score = self._calculate_wellness_score(activity_features)
        
        return SessionFeatures(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            duration_minutes=duration_minutes,
            total_messages=len(messages),
            user_messages=len(user_messages),
            assistant_messages=len(assistant_messages),
            avg_message_length=statistics.mean(message_lengths) if message_lengths else 0,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            total_words=total_words,
            unique_words=unique_words,
            vocabulary_diversity=vocabulary_diversity,
            rare_word_percentage=rare_word_percentage,
            dominant_labels=dominant_labels,
            label_diversity=label_diversity,
            activity_features=activity_features,
            engagement_score=engagement_score,
            productivity_score=productivity_score,
            wellness_score=wellness_score
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple word tokenization
        words = re.findall(r'\b[a-z]+\b', text.lower())
        return words
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _count_messages_in_window(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> int:
        """Count messages in a time window."""
        # Query graph for message nodes in window
        nodes = self.graph.search_nodes(
            node_type="log",
            created_after=start_time.isoformat(),
            created_before=end_time.isoformat()
        )
        
        return len(nodes)
    
    def _get_logs_in_window(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        """Get all logs in a time window."""
        # Query graph for log nodes
        nodes = self.graph.search_nodes(
            node_type="log",
            created_after=start_time.isoformat(),
            created_before=end_time.isoformat()
        )
        
        logs = []
        for node in nodes:
            if "attrs" in node:
                log_data = node["attrs"]
                log_data["timestamp"] = node.get("created_at", datetime.now(timezone.utc).isoformat())
                logs.append(log_data)
        
        return logs
    
    def _calculate_engagement_score(
        self,
        response_times: List[float],
        message_lengths: List[int]
    ) -> float:
        """
        Calculate engagement score based on response patterns.
        
        Quick responses and longer messages indicate higher engagement.
        """
        if not response_times or not message_lengths:
            return 0.0
        
        # Normalize response times (faster = higher score)
        avg_response = statistics.mean(response_times)
        response_score = max(0.0, min(1.0, 1.0 - (avg_response / 300.0)))  # 5 minutes = 0.0
        
        # Normalize message lengths (longer = higher score)
        avg_length = statistics.mean(message_lengths)
        length_score = min(1.0, avg_length / 200)  # 200 chars = 1.0
        
        # Combine scores
        engagement = (response_score * 0.6 + length_score * 0.4)
        
        return engagement

    def _calculate_productivity_score(self, activity: ActivityFeatures) -> float:
        """Calculate productivity score from activity features."""
        if activity.total_logs == 0:
            return 0.0

        components = []

        # Task completion
        if activity.tasks_completed > 0:
            task_score = min(1.0, activity.tasks_completed / 5)  # 5 tasks = 1.0
            components.append(task_score * 0.3)

        # Pomodoro sessions
        if activity.pomodoros_completed > 0:
            pomo_score = min(1.0, activity.pomodoros_completed / 4)  # 4 pomodoros = 1.0
            components.append(pomo_score * 0.3)

        # Focus quality
        if activity.avg_focus_score > 0:
            focus_score = activity.avg_focus_score / 10  # Assuming 10-point scale
            components.append(focus_score * 0.2)

        # Study time
        if activity.study_hours > 0:
            study_score = min(1.0, activity.study_hours / 4.0)  # 4 hours = 1.0
            components.append(study_score * 0.2)

        return sum(components) if components else 0.0

    def _calculate_wellness_score(self, activity: ActivityFeatures) -> float:
        """Calculate wellness score from activity features."""
        if activity.total_logs == 0:
            return 0.0

        components = []

        # Exercise
        if activity.exercise_minutes > 0:
            exercise_score = min(1.0, activity.exercise_minutes / 30.0)  # 30 min = 1.0
            components.append(exercise_score * 0.3)

        # Sleep
        if activity.sleep_hours > 0:
            # Optimal sleep is 7-9 hours
            if 7 <= activity.sleep_hours <= 9:
                sleep_score = 1.0
            elif activity.sleep_hours < 7:
                sleep_score = activity.sleep_hours / 7
            else:
                sleep_score = max(0.0, 1.0 - (activity.sleep_hours - 9.0) / 3.0)
            components.append(sleep_score * 0.3)

        # Mood
        if activity.avg_mood > 0:
            mood_score = activity.avg_mood / 10  # Assuming 10-point scale
            components.append(mood_score * 0.2)

        # Energy
        if activity.avg_energy > 0:
            energy_score = activity.avg_energy / 10  # Assuming 10-point scale
            components.append(energy_score * 0.2)

        return sum(components) if components else 0.0
    
    def save_features(self, features: Any, filename: str) -> None:
        """Save features to disk."""
        filepath = os.path.join(self.data_dir, filename)
        
        if hasattr(features, 'to_dict'):
            data = features.to_dict()
        else:
            data = features
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def save_vocabulary_stats(self) -> None:
        """Save user vocabulary statistics."""
        vocab_file = os.path.join(self.data_dir, "user_vocabulary.json")
        with open(vocab_file, "w") as f:
            json.dump(dict(self.user_vocabulary), f, indent=2)
        
        # Update top 1000 words
        self.top_1000_words = set(
            word for word, _ in self.user_vocabulary.most_common(1000)
        )
    
    def get_feature_summary(self, session_features: SessionFeatures) -> Dict[str, Any]:
        """Get a human-readable summary of features."""
        summary = {
            "session": {
                "duration": f"{session_features.duration_minutes:.1f} minutes",
                "messages": session_features.total_messages,
                "engagement": f"{session_features.engagement_score:.2f}/1.00"
            },
            "vocabulary": {
                "total_words": session_features.total_words,
                "unique_words": session_features.unique_words,
                "diversity": f"{session_features.vocabulary_diversity:.2%}",
                "rare_words": f"{session_features.rare_word_percentage:.1f}%"
            },
            "topics": session_features.dominant_labels[:3],
            "behavioral_scores": {
                "productivity": f"{session_features.productivity_score:.2f}/1.00",
                "wellness": f"{session_features.wellness_score:.2f}/1.00"
            }
        }
        
        if session_features.activity_features:
            activity = session_features.activity_features
            summary["activity"] = {
                "logs": activity.total_logs,
                "categories": activity.unique_categories,
                "tasks": activity.tasks_completed,
                "focus_sessions": activity.focus_sessions,
                "exercise_minutes": activity.exercise_minutes
            }
        
        return summary


# Example usage
if __name__ == "__main__":
    from assistant.logger.unified import UnifiedLogger
    from assistant.graph.store import GraphStore
    
    # Initialize
    logger = UnifiedLogger(data_dir="data/tracking")
    graph = GraphStore(data_dir="data/graph")
    extractor = EnhancedFeatureExtractor(logger, graph)
    
    # Extract features from a message
    message = "I just finished a really productive 2-hour coding session! Got the new feature working and wrote comprehensive tests. Feeling energized and ready to tackle the next challenge."
    
    timestamp = datetime.now(timezone.utc)
    features = extractor.extract_message_features(message, timestamp)
    
    print("Message Features:")
    print(f"  Words: {features.word_count} ({features.unique_words} unique)")
    print(f"  Vocabulary richness: {features.vocabulary_richness:.2f}")
    print(f"  Rare words: {features.rare_word_count} ({features.rare_word_ratio:.2%})")
    print(f"  Questions: {features.question_marks}, Exclamations: {features.exclamation_marks}")
    
    # Extract activity features for last hour
    activity = extractor.extract_activity_features(
        timestamp - timedelta(hours=1),
        timestamp
    )
    
    print(f"\nActivity Features (last hour):")
    print(f"  Total logs: {activity.total_logs}")
    print(f"  Categories: {activity.unique_categories}")
    print(f"  Tasks completed: {activity.tasks_completed}")
    
    # Create mock session for testing
    messages = [
        {"role": "user", "content": message, "timestamp": timestamp.isoformat()},
        {"role": "assistant", "content": "That's fantastic! What feature did you implement?", 
         "timestamp": (timestamp + timedelta(seconds=5)).isoformat()},
        {"role": "user", "content": "The new context-aware scheduling system we discussed.", 
         "timestamp": (timestamp + timedelta(seconds=30)).isoformat()}
    ]
    
    labels = [
        ["productivity", "coding", "achievement"],
        ["encouragement", "question"],
        ["technical", "scheduling", "context"]
    ]
    
    session_features = extractor.extract_session_features(
        "test_session",
        messages,
        labels
    )
    
    # Get summary
    summary = extractor.get_feature_summary(session_features)
    print(f"\nSession Summary:")
    print(json.dumps(summary, indent=2))
    
    # Save features
    extractor.save_features(session_features, "session_features_test.json")
    extractor.save_vocabulary_stats()
    
    print("\nFeatures saved to data/features/")
