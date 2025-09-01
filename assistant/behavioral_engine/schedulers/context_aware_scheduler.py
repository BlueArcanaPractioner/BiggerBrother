"""
Context-Aware Conversational Scheduler with Labeling Integration
================================================================

Integrates with the existing labeling pipeline to store all conversations
as chunks and build context from similar past messages.
"""

from __future__ import annotations
import json
import os
import hashlib
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# Use the app's OpenAI client
from app.openai_client import OpenAIClient

# Import the actual labeling functions (not classes that don't exist)
from app import label_generator
from app import coverage_checker

# Import core assistant components
from assistant.logger.unified import UnifiedLogger
from assistant.graph.store import GraphStore

# Import the ACTUAL harmonizer class name
from assistant.importers.harmonizer_v2 import EnhancedHarmonizer

# Import check-in types
from assistant.conversational_logger import CheckInType


def ensure_timezone_aware(dt):
    """Ensure datetime is timezone-aware. Use UTC if no timezone specified."""
    if dt is None:
        return datetime.now(timezone.utc)
    if not hasattr(dt, 'tzinfo') or dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class ConversationContext:
    """Manages conversation context with similarity-based retrieval."""

    max_chars: int = 100000
    temporal_window_chars: int = 100000
    relevancy_threshold: float = 0.5

    # Separate contexts
    similar_messages: List[Dict] = field(default_factory=list)
    temporal_messages: List[Dict] = field(default_factory=list)

    # Character counts
    similar_chars: int = 0
    temporal_chars: int = 0

    def add_similar_message(self, message: Dict, relevancy: float) -> bool:
        """
        Add a message to similar context if relevant enough.

        Returns:
            True if added, False if rejected (below threshold or over limit)
        """
        if relevancy < self.relevancy_threshold:
            return False

        message_chars = len(json.dumps(message))

        if self.similar_chars + message_chars > self.max_chars:
            return False

        # Add with relevancy score
        message_with_score = {**message, "_relevancy": relevancy}
        self.similar_messages.append(message_with_score)
        self.similar_chars += message_chars

        return True

    def add_temporal_message(self, message: Dict) -> bool:
        """
        Add a message to temporal context (rolling window).

        Returns:
            True if added, False if over limit
        """
        message_chars = len(json.dumps(message))

        # Remove old messages if over limit
        while self.temporal_chars + message_chars > self.temporal_window_chars and self.temporal_messages:
            removed = self.temporal_messages.pop(0)
            self.temporal_chars -= len(json.dumps(removed))

        self.temporal_messages.append(message)
        self.temporal_chars += message_chars

        return True

    def get_context_prompt(self) -> str:
        """Build context prompt for OpenAI."""
        prompt_parts = []

        if self.similar_messages:
            prompt_parts.append("=== Similar Past Conversations ===")
            # Sort by relevancy
            sorted_similar = sorted(
                self.similar_messages,
                key=lambda m: m.get("_relevancy", 0),
                reverse=True
            )
            for msg in sorted_similar[:20]:  # Top 20 most relevant
                prompt_parts.append(f"[Relevancy: {msg.get('_relevancy', 0):.2f}]")
                prompt_parts.append(f"User: {msg.get('user_message', '')}")
                prompt_parts.append(f"Assistant: {msg.get('assistant_response', '')}")
                if msg.get('extracted_data'):
                    prompt_parts.append(f"Extracted: {json.dumps(msg['extracted_data'])}")
                prompt_parts.append("")

        if self.temporal_messages:
            prompt_parts.append("=== Recent Conversation ===")
            for msg in self.temporal_messages[-10:]:  # Last 10 messages
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                prompt_parts.append(f"{role.capitalize()}: {content}")
            prompt_parts.append("")

        return "\n".join(prompt_parts)


@dataclass
class SandboxEnvironment:
    """Isolated environment for RP campaigns or experiments."""

    name: str
    data_dir: str
    labels_dir: str
    chunks_dir: str
    ruleset: Dict[str, Any] = field(default_factory=dict)
    character_sheets: Dict[str, Dict] = field(default_factory=dict)

    def __post_init__(self):
        """Create sandbox directories."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)

        # Save sandbox config with timezone-aware timestamp
        config_file = os.path.join(self.data_dir, "sandbox_config.json")
        with open(config_file, "w") as f:
            json.dump({
                "name": self.name,
                "created_at": ensure_timezone_aware(datetime.now(timezone.utc)).isoformat(),
                "ruleset": self.ruleset
            }, f, indent=2)

    def add_character(self, name: str, sheet: Dict) -> None:
        """Add a character sheet to the sandbox."""
        self.character_sheets[name] = sheet

        # Save character sheet
        char_file = os.path.join(self.data_dir, f"character_{name}.json")
        with open(char_file, "w") as f:
            json.dump(sheet, f, indent=2)

    def get_context_modifier(self) -> str:
        """Get sandbox-specific context modifications."""
        modifiers = []

        if self.ruleset:
            modifiers.append(f"=== Campaign Rules ===")
            for rule, value in self.ruleset.items():
                modifiers.append(f"{rule}: {value}")

        if self.character_sheets:
            modifiers.append(f"\n=== Active Characters ===")
            for name, sheet in self.character_sheets.items():
                modifiers.append(f"{name}: {json.dumps(sheet, indent=2)}")

        return "\n".join(modifiers) if modifiers else ""


class ContextAwareScheduler:
    """
    Scheduler that integrates with the labeling pipeline and builds
    context from similar past conversations.
    """

    def __init__(
        self,
        logger: UnifiedLogger,
        graph_store: GraphStore,
        openai_client: OpenAIClient,
        data_dir: str = "data/context_scheduler",
        labels_dir: str = "labels",
        chunks_dir: str = "data/chunks",
        sandbox_mode: bool = False,
        sandbox_name: Optional[str] = None
    ):
        """Initialize context-aware scheduler."""
        self.tzinfo = None
        self.logger = logger
        self.graph = graph_store
        self.openai_client = openai_client

        # Set up directories based on sandbox mode
        if sandbox_mode and sandbox_name:
            sandbox_base = os.path.join("sandboxes", sandbox_name)
            self.sandbox = SandboxEnvironment(
                name=sandbox_name,
                data_dir=os.path.join(sandbox_base, "data"),
                labels_dir=os.path.join(sandbox_base, "labels"),
                chunks_dir=os.path.join(sandbox_base, "chunks")
            )
            self.data_dir = self.sandbox.data_dir
            self.labels_dir = self.sandbox.labels_dir
            self.chunks_dir = self.sandbox.chunks_dir
        else:
            self.sandbox = None
            self.data_dir = data_dir
            self.labels_dir = labels_dir
            self.chunks_dir = chunks_dir

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)

        # Initialize the harmonizer with the correct class name
        self.harmonizer = EnhancedHarmonizer(
            data_dir=self.data_dir,
            cache_file=os.path.join(self.data_dir, "harmonizer_cache.pkl")
        )

        # Conversation state
        self.current_context = ConversationContext()
        self.current_session_id = None
        self.message_count = 0

        # Temporal weights for relevancy
        self.temporal_weights = {
            "last_hour": 2.0,
            "last_24_hours": 1.5,
            "last_week": 1.2,
            "last_month": 1.0,
            "older": 0.8
        }

    def _detect_check_in_type(self) -> CheckInType:
        """
        Auto-detect appropriate check-in type based on time of day.

        Returns:
            Appropriate CheckInType based on current time
        """
        # Use timezone-aware datetime
        current_hour = ensure_timezone_aware(datetime.now(timezone.utc)).hour

        if 5 <= current_hour < 10:
            return CheckInType.MORNING
        elif 10 <= current_hour < 12:
            return CheckInType.FOCUS
        elif 12 <= current_hour < 14:
            return CheckInType.PERIODIC  # Lunch time
        elif 14 <= current_hour < 17:
            return CheckInType.FOCUS
        elif 17 <= current_hour < 20:
            return CheckInType.PERIODIC  # End of work day
        elif 20 <= current_hour < 23:
            return CheckInType.EVENING
        else:
            # Late night or very early morning
            return CheckInType.PERIODIC

    def start_check_in(
            self,
            check_in_type: Optional[CheckInType] = None,  # Make it Optional
            custom_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Start a check-in session with context building.

        Args:
            check_in_type: Type of check-in (auto-detected if None)
            custom_context: Additional context to include

        Returns:
            Session information with greeting
        """
        # Auto-detect check-in type if not provided
        if check_in_type is None:
            check_in_type = self._detect_check_in_type()
            self.logger.info(f"Auto-detected check-in type: {check_in_type.value}")

        # Generate session ID with timezone-aware timestamp
        now = ensure_timezone_aware(datetime.now(timezone.utc))
        self.current_session_id = f"session_{now.strftime('%Y%m%d_%H%M%S')}"
        self.message_count = 0

        # Reset context
        self.current_context = ConversationContext()

        # Build initial context from similar past check-ins
        self._build_initial_context(check_in_type)

        # Add sandbox context if in sandbox mode
        if self.sandbox:
            sandbox_context = self.sandbox.get_context_modifier()
            if sandbox_context:
                self.current_context.add_temporal_message({
                    "role": "system",
                    "content": sandbox_context
                })

        # Generate greeting based on context
        greeting = self._generate_contextual_greeting(check_in_type, custom_context)

        # Store greeting as first chunk with timezone-aware timestamp
        self._store_message_chunk({
            "session_id": self.current_session_id,
            "message_num": self.message_count,
            "role": "assistant",
            "content": greeting,
            "check_in_type": check_in_type.value,
            "timestamp": ensure_timezone_aware(datetime.now(timezone.utc)).isoformat()
        })

        self.message_count += 1

        return {
            "session_id": self.current_session_id,
            "greeting": greeting,
            "check_in_type": check_in_type.value,
            "context_size": self.current_context.similar_chars + self.current_context.temporal_chars
        }

    def process_message(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message with labeling and context updates.

        Args:
            user_message: The user's message

        Returns:
            Response with extracted data and assistant reply
        """
        if not self.current_session_id:
            return {"error": "No active session. Call start_check_in first."}

        # Store user message as chunk with timezone-aware timestamp
        user_chunk = {
            "session_id": self.current_session_id,
            "message_num": self.message_count,
            "role": "user",
            "content": user_message,
            "timestamp": ensure_timezone_aware(datetime.now(timezone.utc)).isoformat()
        }
        chunk_id = self._store_message_chunk(user_chunk)
        self.message_count += 1

        # Generate labels for the message
        labels = self._generate_and_store_labels(user_message, chunk_id)

        # Find similar messages using harmonized labels
        similar_messages = self._find_similar_messages(labels)

        # Update context with similar messages
        for msg, relevancy in similar_messages:
            self.current_context.add_similar_message(msg, relevancy)

        # Add to temporal context
        self.current_context.add_temporal_message(user_chunk)

        # Extract structured data
        extracted_data = self._extract_structured_data(user_message)

        # Log extracted data with timezone-aware timestamps
        logged_entries = []
        for entry in extracted_data.get("entries", []):
            try:
                if "timestamp" not in entry:
                    entry["timestamp"] = ensure_timezone_aware(datetime.now(timezone.utc)).isoformat()

                node = self.logger.log(entry)
                logged_entries.append(node)
            except Exception as e:
                print(f"Error logging entry: {e}")

        # Generate response with context
        assistant_response = self._generate_contextual_response(
            user_message,
            extracted_data,
            self.current_context
        )

        # Store assistant response as chunk with timezone-aware timestamp
        assistant_chunk = {
            "session_id": self.current_session_id,
            "message_num": self.message_count,
            "role": "assistant",
            "content": assistant_response,
            "timestamp": ensure_timezone_aware(datetime.now(timezone.utc)).isoformat(),
            "extracted_data": extracted_data
        }
        self._store_message_chunk(assistant_chunk)
        self.message_count += 1

        # Add to temporal context
        self.current_context.add_temporal_message(assistant_chunk)

        return {
            "response": assistant_response,
            "extracted_count": len(logged_entries),
            "labels": labels,
            "context_size": self.current_context.similar_chars + self.current_context.temporal_chars,
            "similar_messages_count": len(self.current_context.similar_messages)
        }

    def _build_initial_context(self, check_in_type: Optional[CheckInType]) -> None:
        """Build initial context from past similar check-ins."""
        if check_in_type is None:
            check_in_type = CheckInType.PERIODIC

        search_labels = [check_in_type.value, "check_in", "conversation"]
        past_sessions = self._search_past_sessions(search_labels, limit=10)

        for session in past_sessions:
            session_time_str = session.get("timestamp", ensure_timezone_aware(datetime.now(timezone.utc)).isoformat())

            try:
                # Parse and ensure timezone-aware
                if isinstance(session_time_str, str):
                    session_time = ensure_timezone_aware(datetime.fromisoformat(session_time_str.replace('Z', '+00:00')))
                else:
                    session_time = session_time_str

                # Always ensure timezone awareness
                session_time = ensure_timezone_aware(session_time)

            except (ValueError, AttributeError):
                session_time = ensure_timezone_aware(datetime.now(timezone.utc))

            temporal_weight = self._calculate_temporal_weight(session_time)
            base_relevancy = session.get("relevancy", 0.5)
            weighted_relevancy = base_relevancy * temporal_weight

            self.current_context.add_similar_message(session, weighted_relevancy)

    def _generate_and_store_labels(self, message: str, chunk_id: str) -> List[Dict]:
        """Generate labels for a message and store them."""
        # Create a chunk structure for the label generator
        chunk = {
            "gid": chunk_id,
            "content_text": message,
            "role": "user",
            "conversation_id": self.current_session_id
        }

        # Use the label_generator.main function to generate labels
        inputs = {
            "data_root": self.data_dir,
            "manifest_glob": []  # We'll directly provide the chunk
        }
        outputs = {
            "labels_dir": self.labels_dir
        }

        # Generate labels using the module's functions
        # Since we can't easily call main() with a single chunk,
        # we'll use the underlying logic directly
        prompt_head = label_generator._load_prompt_head(self.data_dir)
        prompt = label_generator._build_prompt(prompt_head, chunk)

        try:
            # Call OpenAI to generate labels
            if hasattr(self.openai_client, "chat"):
                raw = self.openai_client.chat(
                    [{"role": "user", "content": prompt}],
                    model="gpt-5-nano",
                    response_format={"type": "json_object"},
                )
            else:
                raw = self.openai_client.complete(prompt=prompt, model="gpt-5-nano")

            # Parse the response
            payload = label_generator._coerce_json(raw)

            # Process through harmonizer
            harmonized = self.harmonizer.process_label_set(
                {
                    "topic": payload.get("topic", []),
                    "tone": payload.get("tone", []),
                    "intent": payload.get("intent", [])
                },
                message_context=message
            )

            # Flatten harmonized labels for storage
            labels = []
            for category, items in harmonized.items():
                for item in items:
                    labels.append({
                        "category": category,
                        "label": item["canonical"],
                        "original": item["original"],
                        "group": item["group"],
                        "confidence": item["p"]
                    })

            # Store labels with timezone-aware timestamp
            label_file = os.path.join(self.labels_dir, f"{chunk_id}.json")
            with open(label_file, "w") as f:
                json.dump({
                    "chunk_id": chunk_id,
                    "labels": labels,
                    "timestamp": ensure_timezone_aware(datetime.now(timezone.utc)).isoformat(),
                    "raw_labels": payload
                }, f, indent=2)

            return labels

        except Exception as e:
            print(f"Error generating labels: {e}")
            return []

    def _find_similar_messages(self, labels: List[Dict]) -> List[Tuple[Dict, float]]:
        """Find similar messages based on labels."""
        similar_messages = []

        # Get canonical labels per category
        labels_by_category = defaultdict(list)
        for label in labels:
            labels_by_category[label["category"]].append(label["label"])

        # Search through label files
        for label_file in os.listdir(self.labels_dir):
            if not label_file.endswith(".json"):
                continue

            try:
                with open(os.path.join(self.labels_dir, label_file), "r") as f:
                    stored_data = json.load(f)

                stored_labels = stored_data.get("labels", [])

                # Build stored labels by category
                stored_by_category = defaultdict(list)
                for label in stored_labels:
                    stored_by_category[label["category"]].append(label["label"])

                # Calculate similarity
                relevancy = self._calculate_multi_category_similarity(
                    labels_by_category,
                    stored_by_category
                )

                if relevancy > 0.3:  # Minimum threshold
                    # Get the corresponding chunk
                    chunk_id = stored_data.get("chunk_id", label_file.replace(".json", ""))
                    chunk = self._load_chunk(chunk_id)

                    if chunk:
                        # Apply temporal weight with timezone-aware comparison
                        chunk_time_str = chunk.get("timestamp", ensure_timezone_aware(datetime.now(timezone.utc)).isoformat())

                        # Parse the timestamp
                        if isinstance(chunk_time_str, str):
                            chunk_time = ensure_timezone_aware(datetime.fromisoformat(chunk_time_str.replace('Z', '+00:00')))
                        else:
                            chunk_time = chunk_time_str

                        # Ensure timezone awareness
                        chunk_time = ensure_timezone_aware(chunk_time)

                        temporal_weight = self._calculate_temporal_weight(chunk_time)
                        weighted_relevancy = relevancy * temporal_weight

                        similar_messages.append((chunk, weighted_relevancy))

            except Exception as e:
                print(f"Error processing label file {label_file}: {e}")

        # Sort by relevancy and return top matches
        similar_messages.sort(key=lambda x: x[1], reverse=True)
        return similar_messages[:20]

    def _calculate_multi_category_similarity(
        self,
        labels1: Dict[str, List[str]],
        labels2: Dict[str, List[str]]
    ) -> float:
        """Calculate similarity between two label sets across categories."""
        if not labels1 or not labels2:
            return 0.0

        # Weight different categories
        category_weights = {
            "topic": 0.5,
            "intent": 0.3,
            "tone": 0.2
        }

        total_similarity = 0.0
        total_weight = 0.0

        for category, weight in category_weights.items():
            set1 = set(labels1.get(category, []))
            set2 = set(labels2.get(category, []))

            if set1 and set2:
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))

                if union > 0:
                    jaccard = intersection / union
                    total_similarity += jaccard * weight
                    total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_similarity / total_weight

    def _calculate_temporal_weight(self, timestamp: datetime) -> float:
        """Calculate temporal weight based on recency."""
        # Ensure both datetimes are timezone-aware
        now = ensure_timezone_aware(datetime.now(timezone.utc))
        timestamp = ensure_timezone_aware(timestamp)

        # Now safe to compare
        age = now - timestamp

        if age < timedelta(hours=1):
            return self.temporal_weights["last_hour"]
        elif age < timedelta(days=1):
            return self.temporal_weights["last_24_hours"]
        elif age < timedelta(weeks=1):
            return self.temporal_weights["last_week"]
        elif age < timedelta(days=30):
            return self.temporal_weights["last_month"]
        else:
            return self.temporal_weights["older"]

    def _store_message_chunk(self, chunk: Dict) -> str:
        """Store a message as a chunk."""
        # Generate chunk ID
        chunk_id = hashlib.sha256(
            f"{chunk['session_id']}_{chunk['message_num']}".encode()
        ).hexdigest()[:16]

        chunk["gid"] = chunk_id

        # Store chunk
        chunk_file = os.path.join(self.chunks_dir, f"{chunk_id}.json")
        with open(chunk_file, "w") as f:
            json.dump(chunk, f, indent=2)

        # Also append to manifest with timezone-aware timestamp
        manifest_file = os.path.join(self.chunks_dir, "manifest.jsonl")
        with open(manifest_file, "a") as f:
            f.write(json.dumps({"gid": chunk_id, "session_id": chunk["session_id"],
                               "message_num": chunk["message_num"],
                               "timestamp": chunk["timestamp"]}) + "\n")

        return chunk_id

    def _load_chunk(self, chunk_id: str) -> Optional[Dict]:
        """Load a chunk by ID."""
        chunk_file = os.path.join(self.chunks_dir, f"{chunk_id}.json")
        if os.path.exists(chunk_file):
            with open(chunk_file, "r") as f:
                return json.load(f)
        return None

    def _search_past_sessions(self, labels: List[str], limit: int = 10) -> List[Dict]:
        """Search for past sessions with given labels."""
        sessions = []

        # This is simplified - in production, you'd use the graph store
        # or a more sophisticated search mechanism
        manifest_file = os.path.join(self.chunks_dir, "manifest.jsonl")
        if os.path.exists(manifest_file):
            with open(manifest_file, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        chunk = self._load_chunk(entry["gid"])
                        if chunk and chunk.get("role") == "user":
                            sessions.append(chunk)
                            if len(sessions) >= limit:
                                break
                    except:
                        continue

        return sessions

    def _extract_structured_data(self, message: str) -> Dict[str, Any]:
        """Extract structured tracking data from message."""
        extraction_prompt = """Extract structured tracking data from this message.
        Look for: nutrition, exercise, mood, sleep, tasks, energy, focus, social interactions.
        
        Return JSON with structure:
        {
            "entries": [
                {
                    "category": "category_name",
                    "value": value,
                    "confidence": 0.0-1.0,
                    "metadata": {}
                }
            ]
        }
        """

        try:
            response = self.openai_client.chat(
                messages=[
                    {"role": "system", "content": extraction_prompt},
                    {"role": "user", "content": message}
                ],
                model="gpt-4o-mini"
            )

            # Parse response
            content = response.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]

            return json.loads(content)
        except Exception as e:
            print(f"Extraction error: {e}")
            return {"entries": []}

    def _generate_contextual_greeting(
        self,
        check_in_type: CheckInType,
        custom_context: Optional[Dict]
    ) -> str:
        """Generate a contextual greeting."""
        context_prompt = self.current_context.get_context_prompt()

        # Add sandbox context if available
        sandbox_modifier = ""
        if self.sandbox:
            sandbox_modifier = self.sandbox.get_context_modifier()

        system_prompt = f"""You are a supportive check-in assistant.
        Generate a natural greeting for a {check_in_type.value} check-in.
        
        {context_prompt}
        
        {sandbox_modifier}
        
        Keep it brief, friendly, and relevant to the check-in type.
        Reference past patterns if relevant from the context.
        """

        if custom_context:
            system_prompt += f"\nAdditional context: {json.dumps(custom_context)}"

        response = self.openai_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate a greeting for {check_in_type.value} check-in"}
            ],
            model="gpt-4o"
        )

        return response.strip()

    def _generate_contextual_response(
        self,
        user_message: str,
        extracted_data: Dict,
        context: ConversationContext
    ) -> str:
        """Generate response with full context."""
        context_prompt = context.get_context_prompt()

        # Add sandbox context if available
        sandbox_modifier = ""
        if self.sandbox:
            sandbox_modifier = self.sandbox.get_context_modifier()

        system_prompt = f"""You are a supportive check-in assistant having a natural conversation.
        
        {context_prompt}
        
        {sandbox_modifier}
        
        Data extracted from user's message: {json.dumps(extracted_data)}
        
        Respond naturally, acknowledging what they shared.
        If similar past conversations suggest patterns, mention them helpfully.
        Keep responses concise and supportive.
        """

        response = self.openai_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="gpt-4o"
        )

        return response.strip()

    def create_sandbox(
        self,
        name: str,
        ruleset: Optional[Dict] = None,
        character_sheets: Optional[Dict[str, Dict]] = None
    ) -> SandboxEnvironment:
        """
        Create a new sandbox environment for RP or experiments.

        Args:
            name: Sandbox name
            ruleset: Rules for the sandbox
            character_sheets: Initial character sheets

        Returns:
            Created sandbox environment
        """
        sandbox_base = os.path.join("sandboxes", name)
        sandbox = SandboxEnvironment(
            name=name,
            data_dir=os.path.join(sandbox_base, "data"),
            labels_dir=os.path.join(sandbox_base, "labels"),
            chunks_dir=os.path.join(sandbox_base, "chunks"),
            ruleset=ruleset or {}
        )

        # Add character sheets
        if character_sheets:
            for char_name, sheet in character_sheets.items():
                sandbox.add_character(char_name, sheet)

        return sandbox

    def switch_to_sandbox(self, sandbox_name: str) -> None:
        """Switch to a sandbox environment."""
        sandbox_base = os.path.join("sandboxes", sandbox_name)

        if not os.path.exists(sandbox_base):
            raise ValueError(f"Sandbox '{sandbox_name}' does not exist")

        self.sandbox = SandboxEnvironment(
            name=sandbox_name,
            data_dir=os.path.join(sandbox_base, "data"),
            labels_dir=os.path.join(sandbox_base, "labels"),
            chunks_dir=os.path.join(sandbox_base, "chunks")
        )

        # Load sandbox config
        config_file = os.path.join(self.sandbox.data_dir, "sandbox_config.json")
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
                self.sandbox.ruleset = config.get("ruleset", {})

        # Load character sheets
        for char_file in os.listdir(self.sandbox.data_dir):
            if char_file.startswith("character_") and char_file.endswith(".json"):
                char_name = char_file.replace("character_", "").replace(".json", "")
                with open(os.path.join(self.sandbox.data_dir, char_file), "r") as f:
                    self.sandbox.character_sheets[char_name] = json.load(f)

        # Update directories
        self.data_dir = self.sandbox.data_dir
        self.labels_dir = self.sandbox.labels_dir
        self.chunks_dir = self.sandbox.chunks_dir

    def get_session_metrics(self) -> Dict[str, Any]:
        """Get metrics for the current session."""
        return {
            "session_id": self.current_session_id,
            "message_count": self.message_count,
            "context_size": {
                "similar_chars": self.current_context.similar_chars,
                "temporal_chars": self.current_context.temporal_chars,
                "total_chars": self.current_context.similar_chars + self.current_context.temporal_chars
            },
            "similar_messages": len(self.current_context.similar_messages),
            "temporal_messages": len(self.current_context.temporal_messages),
            "sandbox": self.sandbox.name if self.sandbox else None
        }


# Example usage
if __name__ == "__main__":
    from assistant.logger.unified import UnifiedLogger
    from assistant.graph.store import GraphStore
    from app.openai_client import OpenAIClient

    # Initialize components
    logger = UnifiedLogger(data_dir="data/tracking")
    graph = GraphStore(data_dir="data/graph")
    openai_client = OpenAIClient()

    # Create context-aware scheduler
    scheduler = ContextAwareScheduler(
        logger=logger,
        graph_store=graph,
        openai_client=openai_client
    )

    # Example: Create a D&D campaign sandbox
    dnd_sandbox = scheduler.create_sandbox(
        name="forgotten_realms",
        ruleset={
            "system": "D&D 5e",
            "setting": "Forgotten Realms",
            "campaign": "Lost Mine of Phandelver",
            "session": 1
        },
        character_sheets={
            "Aragorn": {
                "class": "Ranger",
                "level": 5,
                "hp": 45,
                "ac": 15,
                "stats": {"STR": 16, "DEX": 14, "CON": 13, "INT": 10, "WIS": 14, "CHA": 8}
            },
            "Gandalf": {
                "class": "Wizard",
                "level": 5,
                "hp": 28,
                "ac": 12,
                "stats": {"STR": 8, "DEX": 10, "CON": 12, "INT": 18, "WIS": 16, "CHA": 14}
            }
        }
    )

    # Switch to sandbox
    scheduler.switch_to_sandbox("forgotten_realms")

    # Start a check-in in sandbox mode
    session = scheduler.start_check_in(
        check_in_type=CheckInType.PERIODIC,
        custom_context={"activity": "D&D session prep"}
    )

    print(f"Started session: {session['session_id']}")
    print(f"Context size: {session['context_size']} chars")
    print(f"Greeting: {session['greeting']}")

    # Process a message
    response = scheduler.process_message(
        "The party just defeated the goblin ambush. Aragorn took 8 damage but found a healing potion."
    )

    print(f"\nResponse: {response['response']}")
    print(f"Labels: {[l['label'] for l in response['labels']]}")
    print(f"Context now has {response['similar_messages_count']} similar messages")

    # Get session metrics
    metrics = scheduler.get_session_metrics()
    print(f"\nSession Metrics:")
    print(f"  Messages: {metrics['message_count']}")
    print(f"  Context: {metrics['context_size']['total_chars']} chars")
    print(f"  Sandbox: {metrics['sandbox']}")