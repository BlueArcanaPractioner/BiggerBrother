"""
Dynamic LogBook System with Category Discovery
===============================================

A system that creates and manages log categories dynamically,
maintains separate log books, and uses dual-model conversation
for detailed activity tracking.
"""

from __future__ import annotations
import json
import os
import csv
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import re

# Use app's OpenAI client
from app.openai_client import OpenAIClient

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
class LogCategory:
    """Represents a category of logs with its own directory and schema."""
    
    name: str
    description: str
    directory: str
    
    # Schema for this category
    fields: Dict[str, str] = field(default_factory=dict)  # field_name -> type
    required_fields: List[str] = field(default_factory=list)
    
    # Tracking
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    entry_count: int = 0
    last_entry: Optional[datetime] = None
    
    # Discovery metadata
    proposed_by: str = "user"  # "user", "ai", "system"
    confidence: float = 1.0
    examples: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "description": self.description,
            "directory": self.directory,
            "fields": self.fields,
            "required_fields": self.required_fields,
            "created_at": self.created_at.isoformat(),
            "entry_count": self.entry_count,
            "last_entry": self.last_entry.isoformat() if self.last_entry else None,
            "proposed_by": self.proposed_by,
            "confidence": self.confidence,
            "examples": self.examples
        }


@dataclass
class LogEntry:
    """A single entry in a log book."""
    
    timestamp: datetime
    category: str
    data: Dict[str, Any]
    
    # Metadata
    extracted_by: str = "manual"  # "manual", "conversation", "quick_log"
    confidence: float = 1.0
    raw_text: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "category": self.category,
            "data": self.data,
            "extracted_by": self.extracted_by,
            "confidence": self.confidence,
            "raw_text": self.raw_text,
            "session_id": self.session_id
        }


class DynamicLogBook:
    """
    Dynamic log book system that creates categories on the fly,
    maintains separate directories, and provides intelligent retrieval.
    """
    
    def __init__(
        self,
        base_dir: str = "data/logbooks",
        openai_client: Optional[OpenAIClient] = None
    ):
        """Initialize the dynamic log book system."""
        self.base_dir = base_dir
        self.openai_client = openai_client or OpenAIClient()
        
        os.makedirs(base_dir, exist_ok=True)
        
        # Load existing categories
        self.categories = self._load_categories()
        
        # Initialize default categories
        if not self.categories:
            self._initialize_default_categories()
        
        # Category discovery cache
        self.discovery_cache = []
        
    def _load_categories(self) -> Dict[str, LogCategory]:
        """Load existing log categories."""
        categories = {}
        
        manifest_file = os.path.join(self.base_dir, "categories.json")
        if os.path.exists(manifest_file):
            with open(manifest_file, "r") as f:
                data = json.load(f)
                for cat_data in data:
                    category = LogCategory(
                        name=cat_data["name"],
                        description=cat_data["description"],
                        directory=cat_data["directory"]
                    )
                    category.fields = cat_data.get("fields", {})
                    category.required_fields = cat_data.get("required_fields", [])
                    category.entry_count = cat_data.get("entry_count", 0)
                    if cat_data.get("last_entry"):
                        category.last_entry = ensure_timezone_aware(datetime.fromisoformat(cat_data["last_entry"]))
                    category.proposed_by = cat_data.get("proposed_by", "system")
                    category.confidence = cat_data.get("confidence", 1.0)
                    category.examples = cat_data.get("examples", [])
                    
                    categories[category.name] = category
        
        return categories
    
    def _initialize_default_categories(self):
        """Create default log categories."""
        defaults = [
            {
                "name": "medications",
                "description": "Medication and supplement tracking",
                "fields": {
                    "medication": "string",
                    "dose": "string",
                    "time": "time",
                    "with_food": "boolean",
                    "notes": "string"
                },
                "required_fields": ["medication", "dose", "time"]
            },
            {
                "name": "meals",
                "description": "Food and nutrition tracking",
                "fields": {
                    "meal_type": "string",
                    "foods": "list",
                    "calories": "number",
                    "protein": "number",
                    "notes": "string"
                },
                "required_fields": ["meal_type", "foods"]
            },
            {
                "name": "exercise",
                "description": "Physical activity and workouts",
                "fields": {
                    "activity": "string",
                    "duration_minutes": "number",
                    "intensity": "string",
                    "distance": "number",
                    "notes": "string"
                },
                "required_fields": ["activity", "duration_minutes"]
            },
            {
                "name": "mood",
                "description": "Mood and emotional state tracking",
                "fields": {
                    "mood_rating": "number",
                    "emotions": "list",
                    "triggers": "list",
                    "notes": "string"
                },
                "required_fields": ["mood_rating"]
            },
            {
                "name": "sleep",
                "description": "Sleep quality and patterns",
                "fields": {
                    "bedtime": "time",
                    "wake_time": "time",
                    "quality": "number",
                    "dreams": "string",
                    "interruptions": "number"
                },
                "required_fields": ["bedtime", "wake_time", "quality"]
            },
            {
                "name": "productivity",
                "description": "Work and task completion",
                "fields": {
                    "tasks_completed": "list",
                    "focus_score": "number",
                    "pomodoros": "number",
                    "blockers": "list",
                    "achievements": "list"
                },
                "required_fields": ["tasks_completed"]
            },
            {
                "name": "symptoms",
                "description": "Health symptoms and observations",
                "fields": {
                    "symptom": "string",
                    "severity": "number",
                    "duration": "string",
                    "possible_triggers": "list",
                    "interventions": "list"
                },
                "required_fields": ["symptom", "severity"]
            }
        ]
        
        for cat_data in defaults:
            self.create_category(
                name=cat_data["name"],
                description=cat_data["description"],
                fields=cat_data["fields"],
                required_fields=cat_data["required_fields"],
                proposed_by="system"
            )
    
    def create_category(
        self,
        name: str,
        description: str,
        fields: Optional[Dict[str, str]] = None,
        required_fields: Optional[List[str]] = None,
        proposed_by: str = "user",
        examples: Optional[List[str]] = None
    ) -> LogCategory:
        """
        Create a new log category with its own directory.
        
        Args:
            name: Category name
            description: What this category tracks
            fields: Field definitions (name -> type)
            required_fields: Required field names
            proposed_by: Who proposed this category
            examples: Example log entries
            
        Returns:
            Created category
        """
        # Sanitize name for directory
        safe_name = re.sub(r'[^a-z0-9_]', '_', name.lower())
        directory = os.path.join(self.base_dir, safe_name)
        
        # Create directory
        os.makedirs(directory, exist_ok=True)
        
        # Create category
        category = LogCategory(
            name=name,
            description=description,
            directory=directory,
            fields=fields or {},
            required_fields=required_fields or [],
            proposed_by=proposed_by,
            examples=examples or []
        )
        
        # Save category
        self.categories[name] = category
        self._save_categories()
        
        # Create initial log files
        self._initialize_log_files(category)
        
        return category
    
    def _initialize_log_files(self, category: LogCategory):
        """Initialize log files for a category."""
        # Create CSV log file
        csv_file = os.path.join(category.directory, f"{category.name}.csv")
        if not os.path.exists(csv_file):
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                # Write header
                headers = ["timestamp", "confidence", "extracted_by"]
                headers.extend(category.fields.keys())
                writer.writerow(headers)
        
        # Create JSON log file for full entries
        json_file = os.path.join(category.directory, f"{category.name}.jsonl")
        if not os.path.exists(json_file):
            # Create empty file
            open(json_file, "a").close()
        
        # Create README
        readme_file = os.path.join(category.directory, "README.md")
        with open(readme_file, "w") as f:
            f.write(f"# {category.name} Log Book\n\n")
            f.write(f"{category.description}\n\n")
            f.write("## Fields\n\n")
            for field, field_type in category.fields.items():
                required = " (required)" if field in category.required_fields else ""
                f.write(f"- **{field}** ({field_type}){required}\n")
            f.write(f"\n## Statistics\n\n")
            f.write(f"- Created: {category.created_at.isoformat()}\n")
            f.write(f"- Entries: {category.entry_count}\n")
    
    def propose_category(self, conversation_text: str) -> Optional[LogCategory]:
        """
        Use GPT-5-nano to propose a new log category from conversation.
        
        Args:
            conversation_text: Recent conversation text
            
        Returns:
            Proposed category or None
        """
        prompt = f"""Analyze this conversation and determine if a new log category would be useful.

Conversation:
{conversation_text}

Current categories: {', '.join(self.categories.keys())}

If a new category would be useful (something being tracked that doesn't fit existing categories), 
propose it with this JSON structure:
{{
    "name": "category_name",
    "description": "what this tracks",
    "fields": {{
        "field1": "type (string/number/boolean/time/date/list)",
        "field2": "type"
    }},
    "required_fields": ["field1"],
    "examples": ["example entry 1", "example entry 2"],
    "confidence": 0.0-1.0
}}

If no new category needed, return: {{"needed": false}}"""

        try:
            response = self.openai_client.chat(
                messages=[
                    {"role": "system", "content": "You are a log categorization assistant."},
                    {"role": "user", "content": prompt}
                ],
                model="gpt-5-nano"  # Fast, efficient model
            )
            
            # Parse response
            content = response.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            
            data = json.loads(content)
            
            if data.get("needed") == False:
                return None
            
            # Create proposed category
            category = self.create_category(
                name=data["name"],
                description=data["description"],
                fields=data.get("fields", {}),
                required_fields=data.get("required_fields", []),
                proposed_by="ai",
                examples=data.get("examples", [])
            )
            
            category.confidence = data.get("confidence", 0.8)
            
            return category
            
        except Exception as e:
            print(f"Error proposing category: {e}")
            return None
    
    def log_entry(
        self,
        category_name: str,
        data: Dict[str, Any],
        raw_text: Optional[str] = None,
        extracted_by: str = "manual",
        confidence: float = 1.0,
        session_id: Optional[str] = None
    ) -> LogEntry:
        """
        Log an entry to a specific category.
        
        Args:
            category_name: Category to log to
            data: Entry data
            raw_text: Original text if extracted
            extracted_by: How this was extracted
            confidence: Confidence in extraction
            session_id: Associated conversation session
            
        Returns:
            Created log entry
        """
        if category_name not in self.categories:
            raise ValueError(f"Category '{category_name}' does not exist")
        
        category = self.categories[category_name]
        
        # Validate required fields
        for field in category.required_fields:
            if field not in data:
                raise ValueError(f"Required field '{field}' missing")
        
        # Create entry
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc),
            category=category_name,
            data=data,
            extracted_by=extracted_by,
            confidence=confidence,
            raw_text=raw_text,
            session_id=session_id
        )
        
        # Save to files
        self._save_entry(category, entry)
        
        # Update category stats
        category.entry_count += 1
        category.last_entry = entry.timestamp
        self._save_categories()
        
        return entry
    
    def _save_entry(self, category: LogCategory, entry: LogEntry):
        """Save entry to category log files."""
        # Save to CSV
        csv_file = os.path.join(category.directory, f"{category.name}.csv")
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            row = [
                entry.timestamp.isoformat(),
                entry.confidence,
                entry.extracted_by
            ]
            for field in category.fields.keys():
                value = entry.data.get(field, "")
                if isinstance(value, list):
                    value = "|".join(str(v) for v in value)
                row.append(value)
            writer.writerow(row)
        
        # Save to JSONL
        json_file = os.path.join(category.directory, f"{category.name}.jsonl")
        with open(json_file, "a") as f:
            f.write(json.dumps(entry.to_dict(), default=str) + "\n")
        
        # Save daily summary
        self._update_daily_summary(category, entry)
    
    def _update_daily_summary(self, category: LogCategory, entry: LogEntry):
        """Update daily summary file."""
        date_str = entry.timestamp.strftime("%Y%m%d")
        summary_file = os.path.join(category.directory, f"daily_{date_str}.json")
        
        # Load existing summary
        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                summary = json.load(f)
        else:
            summary = {
                "date": entry.timestamp.strftime("%Y-%m-%d"),
                "category": category.name,
                "entries": [],
                "stats": {}
            }
        
        # Add entry
        summary["entries"].append(entry.to_dict())
        
        # Update stats
        summary["stats"]["total_entries"] = len(summary["entries"])
        summary["stats"]["confidence_avg"] = sum(
            e["confidence"] for e in summary["entries"]
        ) / len(summary["entries"])
        
        # Save updated summary
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
    
    def get_categories_for_context(self) -> List[Dict]:
        """
        Get available categories for loading into context.
        
        Returns:
            List of category summaries
        """
        summaries = []
        
        for name, category in self.categories.items():
            summaries.append({
                "name": name,
                "description": category.description,
                "directory": category.directory,
                "fields": list(category.fields.keys()),
                "entry_count": category.entry_count,
                "last_entry": category.last_entry.isoformat() if category.last_entry else None,
                "proposed_by": category.proposed_by
            })
        
        return summaries
    
    def load_category_context(
        self,
        category_name: str,
        days_back: int = 7,
        max_entries: int = 50
    ) -> List[Dict]:
        """
        Load recent entries from a category for context.
        
        Args:
            category_name: Category to load
            days_back: How many days to look back
            max_entries: Maximum entries to return
            
        Returns:
            Recent entries from the category
        """
        if category_name not in self.categories:
            return []
        
        category = self.categories[category_name]
        
        # Calculate cutoff date
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        # Load from JSONL
        entries = []
        json_file = os.path.join(category.directory, f"{category_name}.jsonl")
        
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                for line in f:
                    try:
                        entry_data = json.loads(line)
                        entry_time = ensure_timezone_aware(datetime.fromisoformat(entry_data["timestamp"]))
                        
                        if entry_time >= cutoff:
                            entries.append(entry_data)
                    except:
                        continue
        
        # Sort by timestamp and limit
        entries.sort(key=lambda e: e["timestamp"], reverse=True)
        
        return entries[:max_entries]
    
    def search_logs(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[Dict]:
        """
        Search across log books.
        
        Args:
            query: Search query
            categories: Categories to search (None = all)
            date_range: Date range to search
            
        Returns:
            Matching log entries
        """
        results = []
        search_categories = categories or list(self.categories.keys())
        
        for cat_name in search_categories:
            if cat_name not in self.categories:
                continue
            
            category = self.categories[cat_name]
            json_file = os.path.join(category.directory, f"{cat_name}.jsonl")
            
            if not os.path.exists(json_file):
                continue
            
            with open(json_file, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        
                        # Check date range
                        if date_range:
                            entry_time = ensure_timezone_aware(datetime.fromisoformat(entry["timestamp"]))
                            if not (date_range[0] <= entry_time <= date_range[1]):
                                continue
                        
                        # Check query match
                        entry_str = json.dumps(entry).lower()
                        if query.lower() in entry_str:
                            results.append({
                                "category": cat_name,
                                **entry
                            })
                    except:
                        continue
        
        return results
    
    def _save_categories(self):
        """Save category manifest."""
        manifest_file = os.path.join(self.base_dir, "categories.json")
        
        data = [cat.to_dict() for cat in self.categories.values()]
        
        with open(manifest_file, "w") as f:
            json.dump(data, f, indent=2, default=str)


class DetailedActivityLogger:
    """
    Creates detailed activity logs using dual-model conversation:
    GPT-4o for natural conversation and GPT-5-nano for extraction.
    """
    
    def __init__(
        self,
        logbook: DynamicLogBook,
        logger: UnifiedLogger,
        openai_client: Optional[OpenAIClient] = None
    ):
        """Initialize detailed activity logger."""
        self.logbook = logbook
        self.logger = logger
        self.openai_client = openai_client or OpenAIClient()
        
        # Session state
        self.current_session = None
        self.conversation_history = []
        self.extracted_data = []
    
    def start_detailed_logging(self) -> Dict:
        """Start a detailed logging session."""
        self.current_session = {
            "id": f"detailed_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "started_at": datetime.now(timezone.utc),
            "messages": [],
            "extracted": []
        }
        
        # Generate conversational greeting
        greeting = self._generate_greeting()
        
        self.current_session["messages"].append({
            "role": "assistant",
            "content": greeting
        })
        
        return {
            "session_id": self.current_session["id"],
            "greeting": greeting
        }
    
    def process_activity_description(self, description: str) -> Dict:
        """
        Process activity description with dual models.
        
        Args:
            description: Natural language activity description
            
        Returns:
            Response with extracted data
        """
        if not self.current_session:
            return {"error": "No active session"}
        
        # Add to conversation
        self.current_session["messages"].append({
            "role": "user",
            "content": description
        })
        
        # Extract structured data with GPT-5-nano
        extracted = self._extract_activities(description)
        
        # Check if new category needed
        for activity in extracted:
            if activity.get("category") == "unknown":
                # Propose new category
                proposed = self.logbook.propose_category(description)
                if proposed:
                    activity["category"] = proposed.name
        
        # Log extracted activities
        logged = []
        for activity in extracted:
            category = activity.get("category")
            if category and category in self.logbook.categories:
                try:
                    entry = self.logbook.log_entry(
                        category_name=category,
                        data=activity.get("data", {}),
                        raw_text=description,
                        extracted_by="conversation",
                        confidence=activity.get("confidence", 0.8),
                        session_id=self.current_session["id"]
                    )
                    logged.append(category)
                    
                    # Also log to unified logger
                    self.logger.log({
                        "category": category,
                        "value": activity.get("data"),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "confidence": activity.get("confidence", 0.8),
                        "metadata": {"session_id": self.current_session["id"]}
                    })
                except Exception as e:
                    print(f"Error logging to {category}: {e}")
        
        # Generate conversational response with GPT-4o
        response = self._generate_response(description, extracted, logged)
        
        self.current_session["messages"].append({
            "role": "assistant",
            "content": response
        })
        
        self.current_session["extracted"].extend(extracted)
        
        return {
            "response": response,
            "extracted": extracted,
            "logged_to": logged,
            "categories_available": list(self.logbook.categories.keys())
        }
    
    def _generate_greeting(self) -> str:
        """Generate conversational greeting with GPT-4o."""
        # Load context from recent logs
        available_categories = self.logbook.get_categories_for_context()
        
        prompt = f"""You are a friendly activity logger assistant.
        
Available log categories: {', '.join(c['name'] for c in available_categories)}

Generate a warm, conversational greeting for detailed activity logging.
Ask what they've been up to and mention you can help track activities in detail.
Keep it natural and encouraging."""

        response = self.openai_client.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Start a detailed logging session"}
            ],
            model="gpt-4o"  # Conversational model
        )
        
        return response.strip()
    
    def _extract_activities(self, description: str) -> List[Dict]:
        """Extract structured activities with GPT-5-nano."""
        categories_info = json.dumps(
            {name: {"fields": cat.fields, "required": cat.required_fields}
             for name, cat in self.logbook.categories.items()},
            indent=2
        )
        
        prompt = f"""Extract structured activity data from this description.

Description: {description}

Available categories and their schemas:
{categories_info}

Extract all activities mentioned and return as JSON array:
[
    {{
        "category": "category_name",
        "data": {{
            // fields according to category schema
        }},
        "confidence": 0.0-1.0,
        "time_reference": "when this happened"
    }}
]

If an activity doesn't fit existing categories, use "category": "unknown" and suggest fields."""

        try:
            response = self.openai_client.chat(
                messages=[
                    {"role": "system", "content": "You are a precise activity extraction assistant."},
                    {"role": "user", "content": prompt}
                ],
                model="gpt-5-nano"  # Fast extraction model
            )
            
            # Parse response
            content = response.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            
            return json.loads(content)
            
        except Exception as e:
            print(f"Extraction error: {e}")
            return []
    
    def _generate_response(
        self,
        description: str,
        extracted: List[Dict],
        logged: List[str]
    ) -> str:
        """Generate conversational response with GPT-4o."""
        prompt = f"""You are having a natural conversation about someone's activities.

They said: {description}

You extracted and logged:
- {len(extracted)} activities
- Logged to categories: {', '.join(logged) if logged else 'none yet'}

Respond naturally, acknowledging what they shared.
Ask a follow-up question to get more detail about something interesting they mentioned.
Keep it conversational and supportive."""

        response = self.openai_client.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Generate a natural response"}
            ],
            model="gpt-4o"  # Conversational model
        )
        
        return response.strip()
    
    def complete_session(self) -> Dict:
        """Complete the detailed logging session."""
        if not self.current_session:
            return {"error": "No active session"}
        
        self.current_session["completed_at"] = datetime.now(timezone.utc)
        duration = (
            self.current_session["completed_at"] - self.current_session["started_at"]
        ).total_seconds() / 60
        
        # Generate summary
        summary = {
            "session_id": self.current_session["id"],
            "duration_minutes": duration,
            "messages": len(self.current_session["messages"]),
            "activities_extracted": len(self.current_session["extracted"]),
            "categories_used": list(set(
                e["category"] for e in self.current_session["extracted"]
                if e.get("category") != "unknown"
            ))
        }
        
        # Save session
        session_file = os.path.join(
            self.logbook.base_dir,
            f"sessions/{self.current_session['id']}.json"
        )
        os.makedirs(os.path.dirname(session_file), exist_ok=True)
        
        with open(session_file, "w") as f:
            json.dump(self.current_session, f, indent=2, default=str)
        
        self.current_session = None
        
        return summary


class LogBookAssistant:
    """
    Assistant that can retrieve and reason about log books.
    """
    
    def __init__(
        self,
        logbook: DynamicLogBook,
        openai_client: Optional[OpenAIClient] = None
    ):
        """Initialize log book assistant."""
        self.logbook = logbook
        self.openai_client = openai_client or OpenAIClient()
    
    def decide_context_needed(self, query: str) -> List[str]:
        """
        Decide which log categories to load for context.
        
        Args:
            query: User query
            
        Returns:
            List of category names to load
        """
        categories = self.logbook.get_categories_for_context()
        
        prompt = f"""Given this query, decide which log categories would be helpful to load.

Query: {query}

Available categories:
{json.dumps(categories, indent=2)}

Return a JSON array of category names that would be relevant.
Only include categories that would actually help answer the query.
Return empty array if no logs needed.

Example: ["medications", "symptoms"]"""

        try:
            response = self.openai_client.chat(
                messages=[
                    {"role": "system", "content": "You are a log retrieval assistant."},
                    {"role": "user", "content": prompt}
                ],
                model="gpt-5-nano"  # Fast decision model
            )
            
            # Parse response
            content = response.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            
            return json.loads(content)
            
        except Exception as e:
            print(f"Context decision error: {e}")
            return []
    
    def answer_with_logs(self, query: str) -> str:
        """
        Answer a query using relevant log books.
        
        Args:
            query: User query
            
        Returns:
            Answer with log context
        """
        # Decide which logs to load
        categories_needed = self.decide_context_needed(query)
        
        # Load relevant logs
        context = {}
        for category in categories_needed:
            entries = self.logbook.load_category_context(
                category,
                days_back=30,
                max_entries=20
            )
            if entries:
                context[category] = entries
        
        # Generate answer with context
        prompt = f"""Answer this query using the log data provided.

Query: {query}

Relevant log data:
{json.dumps(context, indent=2) if context else "No relevant logs found"}

Provide a helpful, specific answer based on the actual log data.
Cite specific dates and values when relevant.
If the logs don't contain enough information, say so."""

        response = self.openai_client.chat(
            messages=[
                {"role": "system", "content": "You are a helpful assistant with access to personal logs."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-4o"
        )
        
        return response.strip()


# Example usage
if __name__ == "__main__":
    from assistant.logger.unified import UnifiedLogger
    from assistant.graph.store import GraphStore
    from app.openai_client import OpenAIClient
    
    # Initialize
    logger = UnifiedLogger(data_dir="data/tracking")
    graph = GraphStore(data_dir="data/graph")
    openai_client = OpenAIClient()
    
    # Create log book system
    logbook = DynamicLogBook(
        base_dir="data/logbooks",
        openai_client=openai_client
    )
    
    # Create detailed activity logger
    activity_logger = DetailedActivityLogger(
        logbook=logbook,
        logger=logger,
        openai_client=openai_client
    )
    
    # Create assistant
    assistant = LogBookAssistant(
        logbook=logbook,
        openai_client=openai_client
    )
    
    print("📚 Dynamic LogBook System")
    print("=" * 40)
    
    # Show available categories
    print("\nAvailable Log Categories:")
    for cat in logbook.get_categories_for_context():
        print(f"  📁 {cat['name']}: {cat['description']} ({cat['entry_count']} entries)")
    
    # Start detailed logging session
    print("\n--- Starting Detailed Activity Logging ---")
    session = activity_logger.start_detailed_logging()
    print(f"Assistant: {session['greeting']}")
    
    # Example detailed activity description
    user_input = """
    This morning I woke up at 6:30am feeling pretty good, about a 7/10 energy.
    Had breakfast - oatmeal with berries and a protein shake, probably 450 calories.
    Took my morning meds: L-theanine 200mg with coffee, vitamin D 5000IU, and magnesium 400mg.
    Then did a 30 minute run at moderate intensity, covered about 3 miles.
    Noticed slight knee pain during the run, maybe a 3/10 severity.
    """
    
    print(f"\nYou: {user_input}")
    
    result = activity_logger.process_activity_description(user_input)
    print(f"\nAssistant: {result['response']}")
    print(f"\nExtracted {len(result['extracted'])} activities")
    print(f"Logged to: {', '.join(result['logged_to'])}")
    
    # Complete session
    summary = activity_logger.complete_session()
    print(f"\nSession complete: {summary['activities_extracted']} activities logged")
    
    # Test category proposal
    print("\n--- Testing Category Discovery ---")
    
    test_conversation = """
    I've been tracking my guitar practice sessions. Today I practiced scales 
    for 20 minutes, then worked on the solo from Stairway to Heaven for 30 minutes.
    My fingering is getting better but still struggling with the bends.
    """
    
    proposed = logbook.propose_category(test_conversation)
    if proposed:
        print(f"New category proposed: {proposed.name}")
        print(f"Description: {proposed.description}")
        print(f"Fields: {list(proposed.fields.keys())}")
    
    # Test log retrieval
    print("\n--- Testing Log Retrieval ---")
    
    query = "What medications did I take today and at what times?"
    print(f"\nQuery: {query}")
    
    # Assistant decides what to load
    categories_to_load = assistant.decide_context_needed(query)
    print(f"Loading categories: {categories_to_load}")
    
    # Get answer with logs
    answer = assistant.answer_with_logs(query)
    print(f"\nAnswer: {answer}")
    
    # Manual log entry
    print("\n--- Manual Log Entry ---")
    
    entry = logbook.log_entry(
        category_name="medications",
        data={
            "medication": "Melatonin",
            "dose": "3mg",
            "time": "22:00",
            "with_food": False,
            "notes": "For sleep"
        },
        extracted_by="manual"
    )
    
    print(f"✅ Logged to {entry.category} at {entry.timestamp.strftime('%H:%M')}")
    
    # Search logs
    print("\n--- Searching Logs ---")
    
    results = logbook.search_logs(
        query="L-theanine",
        categories=["medications"]
    )
    
    print(f"Found {len(results)} entries mentioning L-theanine")
    for result in results[:3]:
        print(f"  - {result['timestamp']}: {result['data']}")
