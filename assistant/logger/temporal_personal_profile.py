"""
TEMPORAL PERSONAL PROFILE SYSTEM
Maintains evolving personal context with privacy protection and temporal snapshots
"""

import json
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import re
import logging

logger = logging.getLogger(__name__)

def ensure_timezone_aware(dt):
    """Ensure datetime is timezone-aware."""
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class ProfileCategory(Enum):
    """Categories of profile information."""
    EMPLOYMENT = "employment"
    HEALTH = "health"
    RELATIONSHIPS = "relationships"
    ROUTINES = "routines"
    PREFERENCES = "preferences"
    ACTIVITIES = "activities"
    SKILLS = "skills"
    GOALS = "goals"
    CONTEXT = "context"


class PrivacyLevel(Enum):
    """Privacy levels for information."""
    PUBLIC = "public"           # Can be shared freely
    PERSONAL = "personal"       # About the user
    SENSITIVE = "sensitive"     # Health, mental state, etc.
    THIRD_PARTY = "third_party" # About other people - needs anonymization


@dataclass
class ProfileFact:
    """A single fact in the profile."""
    id: str
    category: ProfileCategory
    key: str
    value: Any
    confidence: float
    privacy_level: PrivacyLevel
    source_messages: List[str]
    first_observed: datetime
    last_confirmed: datetime
    update_count: int = 1
    supersedes: Optional[str] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self, current_time: datetime) -> bool:
        """Check if fact has expired."""
        if self.expires_at and current_time > self.expires_at:
            return True
        # Some facts naturally expire
        if self.key in ["current_activity", "immediate_goal", "today_schedule"]:
            age = (current_time - self.last_confirmed).total_seconds() / 3600
            return age > 24  # 24 hours for temporary facts
        return False
    
    def decay_confidence(self, current_time: datetime) -> float:
        """Calculate decayed confidence based on age."""
        days_old = (current_time - self.last_confirmed).days
        # Different decay rates for different categories
        decay_rates = {
            ProfileCategory.EMPLOYMENT: 0.99,    # Slow decay
            ProfileCategory.HEALTH: 0.95,         # Medium decay
            ProfileCategory.ACTIVITIES: 0.90,     # Faster decay
            ProfileCategory.CONTEXT: 0.85,        # Fastest decay
        }
        rate = decay_rates.get(self.category, 0.95)
        return self.confidence * (rate ** days_old)


@dataclass
class PersonEntity:
    """Represents a person mentioned in conversations (anonymized)."""
    anonymous_id: str  # Hash of identifying information
    relationship: Optional[str] = None  # "friend", "colleague", "family"
    first_mentioned: Optional[datetime] = None
    last_mentioned: Optional[datetime] = None
    mention_count: int = 0
    associated_activities: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class ProfileSnapshot:
    """A snapshot of the profile at a point in time."""
    timestamp: datetime
    facts: Dict[str, ProfileFact]
    version: int
    trigger: str  # What caused this snapshot
    stats: Dict[str, Any] = field(default_factory=dict)


class TemporalPersonalProfile:
    """
    Maintains an evolving personal profile with temporal snapshots and privacy protection.
    """
    
    # Patterns for extracting profile facts
    PROFILE_PATTERNS = {
        "employment": [
            (r"(?:work|employed) (?:at|for) ([A-Z][A-Za-z\s&]+)", "employer"),
            (r"(?:I'm|I am) (?:a|an) ([a-z\s]+) (?:at|for)", "job_title"),
            (r"(?:my|our) (?:team|department) (?:is|does) ([^.]+)", "team_info"),
            (r"(\d+) hours? (?:per|a) (?:week|day)", "work_hours"),
        ],
        "health": [
            (r"diagnosed with ([a-z\s]+)", "condition"),
            (r"taking ([a-z]+) for", "medication"),
            (r"(?:sleep|slept) (?:for )?(\d+) hours?", "sleep_duration"),
            (r"(?:feeling|felt) (tired|energetic|exhausted|great)", "energy_level"),
        ],
        "routines": [
            (r"(?:wake up|get up) (?:at|around) (\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm))", "wake_time"),
            (r"(?:go to bed|sleep) (?:at|around) (\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm))", "sleep_time"),
            (r"(?:gym|workout|exercise) (?:every|on) ([A-Za-z\s,]+)", "exercise_schedule"),
        ],
        "relationships": [
            (r"(?:my|our) (wife|husband|partner|spouse)", "partner"),
            (r"(?:my|our) (son|daughter|child|kids?)", "children"),
            (r"(?:living|live) with ([a-z\s]+)", "living_situation"),
        ]
    }
    
    # Facts that should be redacted/anonymized
    SENSITIVE_PATTERNS = [
        r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",  # Full names
        r"\b\d{3}-\d{3}-\d{4}\b",         # Phone numbers
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Emails
        r"\b\d{1,5}\s[A-Z][a-z]+\s(?:St|Ave|Rd|Blvd|Dr|Lane|Way)\b",  # Addresses
    ]
    
    def __init__(
        self,
        data_dir: str = "data/profiles",
        max_facts: int = 1000,
        snapshot_interval_days: int = 7,
        anonymize_third_parties: bool = True
    ):
        """Initialize the profile system."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_facts = max_facts
        self.snapshot_interval = snapshot_interval_days
        self.anonymize_third_parties = anonymize_third_parties
        
        # Current profile state
        self.facts: Dict[str, ProfileFact] = {}
        self.people: Dict[str, PersonEntity] = {}  # Anonymous ID -> Entity
        self.snapshots: List[ProfileSnapshot] = []
        
        # Caching for efficiency
        self.processed_messages: Set[str] = set()
        self.fact_index: Dict[str, List[str]] = defaultdict(list)  # category -> fact_ids
        
        # Load existing profile
        self.load_profile()

    def get_facts_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all facts for a specific category."""
        if isinstance(category, str):
            # Convert string to ProfileCategory enum
            try:
                cat_enum = ProfileCategory(category)
            except ValueError:
                # If category string doesn't match enum values, return empty list
                return []
        else:
            cat_enum = category

        # Return facts as dicts for the requested category
        facts = []
        for fact_id in self.fact_index.get(cat_enum.value, []):
            if fact_id in self.facts:
                fact = self.facts[fact_id]
                facts.append({
                    "id": fact.id,
                    "category": fact.category.value,
                    "key": fact.key,
                    "value": fact.value,
                    "fact": f"{fact.key}: {fact.value}",  # Formatted fact string
                    "confidence": fact.confidence,
                    "last_confirmed": fact.last_confirmed.isoformat()
                })
        return facts
    
    def extract_facts_from_message(
        self,
        message: Dict[str, Any],
        current_time: datetime
    ) -> List[ProfileFact]:
        """Extract profile facts from a message."""
        if message.get('sender') != 'user':
            return []
        
        message_id = message.get('id', str(hash(message.get('content', ''))))
        
        # Skip if already processed
        if message_id in self.processed_messages:
            return []
        
        content = message.get('content', '')
        facts = []
        
        # Anonymize third parties first if needed
        if self.anonymize_third_parties:
            content, person_entities = self._anonymize_content(content, current_time)
            # Update person entities
            for entity in person_entities:
                if entity.anonymous_id not in self.people:
                    self.people[entity.anonymous_id] = entity
                else:
                    self.people[entity.anonymous_id].mention_count += 1
                    self.people[entity.anonymous_id].last_mentioned = current_time
        
        # Extract facts by category
        for category_name, patterns in self.PROFILE_PATTERNS.items():
            category = ProfileCategory(category_name)
            
            for pattern, fact_key in patterns:
                matches = re.finditer(pattern, content.lower())
                for match in matches:
                    value = match.group(1) if match.groups() else match.group(0)
                    
                    # Determine privacy level
                    privacy = self._determine_privacy_level(category, fact_key, value)
                    
                    fact = ProfileFact(
                        id=self._generate_fact_id(category, fact_key, value),
                        category=category,
                        key=fact_key,
                        value=value,
                        confidence=0.7,  # Base confidence
                        privacy_level=privacy,
                        source_messages=[message_id],
                        first_observed=current_time,
                        last_confirmed=current_time
                    )
                    
                    facts.append(fact)
        
        # Mark message as processed
        self.processed_messages.add(message_id)
        
        return facts
    
    def update_profile(
        self,
        new_facts: List[ProfileFact],
        current_time: datetime
    ) -> Dict[str, Any]:
        """Update profile with new facts."""
        stats = {
            "new_facts": 0,
            "updated_facts": 0,
            "superseded_facts": 0,
            "expired_facts": 0
        }
        
        # Remove expired facts
        expired = []
        for fact_id, fact in list(self.facts.items()):
            if fact.is_expired(current_time):
                expired.append(fact_id)
                stats["expired_facts"] += 1
        
        for fact_id in expired:
            del self.facts[fact_id]
        
        # Process new facts
        for new_fact in new_facts:
            existing = self._find_existing_fact(new_fact)
            
            if existing:
                # Update existing fact
                if new_fact.value != existing.value:
                    # Value changed - supersede old fact
                    new_fact.supersedes = existing.id
                    new_fact.first_observed = existing.first_observed
                    new_fact.update_count = existing.update_count + 1
                    self.facts[new_fact.id] = new_fact
                    stats["superseded_facts"] += 1
                else:
                    # Reinforce existing fact
                    existing.last_confirmed = current_time
                    existing.confidence = min(existing.confidence + 0.1, 1.0)
                    existing.source_messages.extend(new_fact.source_messages)
                    existing.update_count += 1
                    stats["updated_facts"] += 1
            else:
                # Add new fact
                self.facts[new_fact.id] = new_fact
                self.fact_index[new_fact.category.value].append(new_fact.id)
                stats["new_facts"] += 1
        
        # Check if snapshot needed
        if self._should_snapshot(current_time):
            self.create_snapshot(current_time, "periodic_update")
        
        return stats
    
    def _anonymize_content(
        self,
        content: str,
        current_time: datetime
    ) -> Tuple[str, List[PersonEntity]]:
        """Anonymize third-party information in content."""
        anonymized = content
        entities = []
        
        # Find potential names and sensitive info
        for pattern in self.SENSITIVE_PATTERNS:
            matches = re.finditer(pattern, content)
            for match in matches:
                original = match.group(0)
                
                # Skip common words that might match patterns
                if original.lower() in ["i", "me", "my", "the", "a", "an"]:
                    continue
                
                # Generate anonymous ID
                anon_id = self._hash_text(original)[:8]
                replacement = f"[PERSON_{anon_id}]"
                
                # Create entity if it looks like a person's name
                if re.match(r"[A-Z][a-z]+ [A-Z][a-z]+", original):
                    entity = PersonEntity(
                        anonymous_id=anon_id,
                        first_mentioned=current_time,
                        last_mentioned=current_time,
                        mention_count=1
                    )
                    entities.append(entity)
                
                anonymized = anonymized.replace(original, replacement)
        
        return anonymized, entities
    
    def _determine_privacy_level(
        self,
        category: ProfileCategory,
        key: str,
        value: str
    ) -> PrivacyLevel:
        """Determine privacy level for a fact."""
        # Health information is always sensitive
        if category == ProfileCategory.HEALTH:
            return PrivacyLevel.SENSITIVE
        
        # Employment details are personal
        if category == ProfileCategory.EMPLOYMENT:
            return PrivacyLevel.PERSONAL
        
        # Check if value contains anonymized references
        if "[PERSON_" in str(value):
            return PrivacyLevel.THIRD_PARTY
        
        # Default to personal
        return PrivacyLevel.PERSONAL
    
    def get_context_for_inference(
        self,
        max_chars: int = 150000,
        categories: Optional[List[ProfileCategory]] = None,
        privacy_filter: Optional[PrivacyLevel] = None
    ) -> str:
        """Get profile context for inference, respecting character limit."""
        context_parts = []
        char_count = 0
        
        # Header
        header = "=== PERSONAL PROFILE CONTEXT ===\n"
        context_parts.append(header)
        char_count += len(header)
        
        # Sort facts by relevance (confidence * recency)
        current_time = datetime.now(timezone.utc)
        sorted_facts = sorted(
            self.facts.values(),
            key=lambda f: f.decay_confidence(current_time),
            reverse=True
        )
        
        # Add facts by category
        for category in (categories or list(ProfileCategory)):
            category_facts = [
                f for f in sorted_facts 
                if f.category == category
                and (privacy_filter is None or f.privacy_level.value <= privacy_filter.value)
            ]
            
            if not category_facts:
                continue
            
            section = f"\n{category.value.upper()}:\n"
            if char_count + len(section) > max_chars:
                break
            
            context_parts.append(section)
            char_count += len(section)
            
            for fact in category_facts:
                fact_str = f"  - {fact.key}: {fact.value} [confidence: {fact.confidence:.1%}]\n"
                
                if char_count + len(fact_str) > max_chars:
                    break
                
                context_parts.append(fact_str)
                char_count += len(fact_str)
        
        # Add recent activity patterns if room
        if char_count < max_chars * 0.8:
            patterns = self._extract_patterns()
            patterns_section = "\nPATTERNS:\n"
            context_parts.append(patterns_section)
            char_count += len(patterns_section)
            
            for pattern_type, pattern_value in patterns.items():
                pattern_str = f"  - {pattern_type}: {pattern_value}\n"
                if char_count + len(pattern_str) > max_chars:
                    break
                context_parts.append(pattern_str)
                char_count += len(pattern_str)
        
        return "".join(context_parts)
    
    def _extract_patterns(self) -> Dict[str, Any]:
        """Extract behavioral patterns from facts."""
        patterns = {}
        
        # Sleep pattern
        wake_times = [
            f for f in self.facts.values() 
            if f.key == "wake_time" and f.category == ProfileCategory.ROUTINES
        ]
        if wake_times:
            avg_wake = sum(self._parse_time(f.value) for f in wake_times) / len(wake_times)
            patterns["typical_wake_time"] = f"{int(avg_wake)}:00"
        
        # Work pattern
        work_facts = [
            f for f in self.facts.values()
            if f.category == ProfileCategory.EMPLOYMENT
        ]
        if work_facts:
            patterns["employment_status"] = "employed"
            employer = next((f.value for f in work_facts if f.key == "employer"), None)
            if employer:
                patterns["employer"] = employer
        
        # Activity frequency
        activities = defaultdict(int)
        for fact in self.facts.values():
            if fact.category == ProfileCategory.ACTIVITIES:
                activities[fact.value] += 1
        
        if activities:
            patterns["frequent_activities"] = dict(
                sorted(activities.items(), key=lambda x: x[1], reverse=True)[:5]
            )
        
        return patterns
    
    def create_snapshot(self, timestamp: datetime, trigger: str) -> ProfileSnapshot:
        """Create a snapshot of the current profile state."""
        snapshot = ProfileSnapshot(
            timestamp=timestamp,
            facts=dict(self.facts),  # Deep copy
            version=len(self.snapshots) + 1,
            trigger=trigger,
            stats={
                "total_facts": len(self.facts),
                "categories": dict(
                    (cat.value, len([f for f in self.facts.values() if f.category == cat]))
                    for cat in ProfileCategory
                ),
                "people_tracked": len(self.people),
                "avg_confidence": sum(f.confidence for f in self.facts.values()) / max(len(self.facts), 1)
            }
        )
        
        self.snapshots.append(snapshot)
        
        # Keep only recent snapshots
        max_snapshots = 50
        if len(self.snapshots) > max_snapshots:
            self.snapshots = self.snapshots[-max_snapshots:]
        
        return snapshot
    
    def _should_snapshot(self, current_time: datetime) -> bool:
        """Check if a snapshot should be created."""
        if not self.snapshots:
            return True
        
        last_snapshot = self.snapshots[-1]
        days_since = (current_time - last_snapshot.timestamp).days
        
        return days_since >= self.snapshot_interval
    
    def _find_existing_fact(self, new_fact: ProfileFact) -> Optional[ProfileFact]:
        """Find existing fact that matches the new one."""
        for fact in self.facts.values():
            if (fact.category == new_fact.category and 
                fact.key == new_fact.key):
                return fact
        return None
    
    def _generate_fact_id(self, category: ProfileCategory, key: str, value: str) -> str:
        """Generate unique ID for a fact."""
        content = f"{category.value}:{key}:{value}"
        return self._hash_text(content)[:16]
    
    def _hash_text(self, text: str) -> str:
        """Generate hash of text for anonymization."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def _parse_time(self, time_str: str) -> float:
        """Parse time string to hours (simplified)."""
        # This is a simplified parser
        match = re.search(r"(\d{1,2})", time_str)
        if match:
            hour = int(match.group(1))
            if "pm" in time_str.lower() and hour < 12:
                hour += 12
            return float(hour)
        return 0.0

    def save_profile(self):
        """Save profile to disk."""
        save_data = {
            "facts": [
                {
                    "id": f.id,
                    "category": f.category.value,
                    "key": f.key,
                    "value": f.value,
                    "confidence": f.confidence,
                    "privacy_level": f.privacy_level.value,
                    "source_messages": f.source_messages,
                    # Ensure datetime objects are converted to ISO format strings
                    "first_observed": f.first_observed.isoformat() if isinstance(f.first_observed,
                                                                                 datetime) else f.first_observed,
                    "last_confirmed": f.last_confirmed.isoformat() if isinstance(f.last_confirmed,
                                                                                 datetime) else f.last_confirmed,
                    "update_count": f.update_count
                }
                for f in self.facts.values()
            ],
            "people": [
                {
                    "anonymous_id": p.anonymous_id,
                    "relationship": p.relationship,
                    "mention_count": p.mention_count,
                    # Handle optional datetime fields
                    "first_mentioned": p.first_mentioned.isoformat() if p.first_mentioned and isinstance(
                        p.first_mentioned, datetime) else p.first_mentioned,
                    "last_mentioned": p.last_mentioned.isoformat() if p.last_mentioned and isinstance(p.last_mentioned,
                                                                                                      datetime) else p.last_mentioned
                }
                for p in self.people.values()
            ],
            "processed_messages": list(self.processed_messages)[-1000:],
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

        save_file = self.data_dir / "profile.json"
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    def load_profile(self):
        """Load profile from disk."""
        save_file = self.data_dir / "profile.json"
        if save_file.exists():
            try:
                with open(save_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load facts
                for item in data.get("facts", []):
                    fact = ProfileFact(
                        id=item["id"],
                        category=ProfileCategory(item["category"]),
                        key=item["key"],
                        value=item["value"],
                        confidence=item["confidence"],
                        privacy_level=PrivacyLevel(item["privacy_level"]),
                        source_messages=item["source_messages"],
                        first_observed=ensure_timezone_aware(datetime.fromisoformat(item["first_observed"])),
                        last_confirmed=ensure_timezone_aware(datetime.fromisoformat(item["last_confirmed"])),
                        update_count=item.get("update_count", 1)
                    )
                    self.facts[fact.id] = fact
                    self.fact_index[fact.category.value].append(fact.id)
                
                # Load people
                for item in data.get("people", []):
                    entity = PersonEntity(
                        anonymous_id=item["anonymous_id"],
                        relationship=item.get("relationship"),
                        mention_count=item.get("mention_count", 0)
                    )
                    self.people[entity.anonymous_id] = entity
                
                # Load processed messages
                self.processed_messages = set(data.get("processed_messages", []))
                
                logger.info(f"Loaded profile with {len(self.facts)} facts and {len(self.people)} people")
                
            except Exception as e:
                logger.error(f"Failed to load profile: {e}")
