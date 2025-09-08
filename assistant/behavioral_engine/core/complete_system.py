"""
Complete Integrated System with Dynamic LogBooks and Two-Tier Harmonization
===========================================================================

Ties together all components including the dynamic log book system
and enhanced label harmonization with simplified two-tier context building.

UPDATED VERSION: Now uses harmonization groups as fast lookup tables for
context selection without needing embeddings at runtime.
"""

from __future__ import annotations
import json
import os
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Counter
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import dotenv

# Use app's OpenAI client (with API quirks handling)
from app.openai_client import OpenAIClient
from app.label_integration_wrappers import LabelGenerator

# Core assistant components
from assistant.logger.unified import UnifiedLogger
from assistant.graph.store import GraphStore
from assistant.logger.temporal_personal_profile import TemporalPersonalProfile
from assistant.conversational_logger import CheckInType

from assistant.behavioral_engine.schedulers.adaptive_scheduler import AdaptiveScheduler
from assistant.behavioral_engine.schedulers.context_aware_scheduler import ContextAwareScheduler
from assistant.behavioral_engine.schedulers.schedule_bridge import ScheduleBridge
from zoneinfo import ZoneInfo

# Import the EnhancedLabelHarmonizer from enhanced_smart_label_importer
from assistant.importers.enhanced_smart_label_importer import EnhancedLabelHarmonizer

# Import all our new components
from assistant.behavioral_engine.features.enhanced_feature_extraction import EnhancedFeatureExtractor
from assistant.behavioral_engine.gamification.rpg_system import RPGSystem
from assistant.behavioral_engine.context.similarity_matcher import ContextSimilarityMatcher
from assistant.behavioral_engine.routines.routine_builder import RoutineBuilder
from assistant.behavioral_engine.logbooks.dynamic_logbook_system import (
    DynamicLogBook,
    DetailedActivityLogger,
    LogBookAssistant
)
from assistant.behavioral_engine.intents.intent_dispatcher import handle_incoming_text

dotenv.load_dotenv()


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


class CompleteIntegratedSystem:
    """
    The complete behavioral intelligence system with all components integrated.
    Now with two-tier harmonization for intelligent context building.
    Uses harmonization groups as fast lookup tables without embeddings.
    """

    def __init__(
            self,
            base_dir: str = None,  # Optional, will use config if not provided
            use_sandbox: bool = False,
            sandbox_name: Optional[str] = None,
            use_config: bool = True  # New parameter to use centralized config
    ):
        """
        Initialize the complete system with enhanced label harmonization and dual schedulers.
        UPDATED VERSION: Two-tier harmonization with fast lookups

        Args:
            base_dir: Base directory for data (optional if use_config=True)
            use_sandbox: Whether to use sandbox mode
            sandbox_name: Name of sandbox if using sandbox mode
            use_config: Use centralized data_config.py paths (default: True)
        """
        if use_config:
            # Import config (do this locally to avoid circular imports)
            from data_config import (
                CHUNKS_DIR, LABELS_DIR, HARMONIZER_DIR, EMBEDDING_CACHE_FILE,
                SCHEDULER_DIR, LOGBOOKS_DIR, FEATURES_DIR, RPG_DIR, ROUTINES_DIR,
                TRACKING_DIR, GRAPH_DIR, PLANNER_DIR, ORCHESTRATOR_DIR,
                PROFILES_DIR, create_missing_directories
            )

            # Create only missing directories
            created, existed = create_missing_directories()

            if created:
                print(f"✅ Created {len(created)} missing directories")

            # Use configured paths
            self.base_dir = str(Path("C:/BiggerBrother-minimal"))  # For compatibility
            chunks_dir = str(CHUNKS_DIR)
            labels_dir = str(LABELS_DIR)
            harmonizer_dir = str(HARMONIZER_DIR)
            # Use the existing embedding cache in data directory
            harmonizer_cache = str(EMBEDDING_CACHE_FILE)  # This is data/embedding_cache.pkl from config
            scheduler_dir = str(SCHEDULER_DIR)
            logbooks_dir = str(LOGBOOKS_DIR)
            features_dir = str(FEATURES_DIR)
            rpg_dir = str(RPG_DIR)
            routines_dir = str(ROUTINES_DIR)
            tracking_dir = str(TRACKING_DIR)
            graph_dir = str(GRAPH_DIR)
            planner_dir = str(PLANNER_DIR)
            orchestrator_dir = str(ORCHESTRATOR_DIR)
            profiles_dir = str(PROFILES_DIR)

        else:
            # Use traditional base_dir approach (backward compatibility)
            self.base_dir = base_dir or "data"
            chunks_dir = f"{self.base_dir}/chunks"
            labels_dir = f"{self.base_dir}/labels"
            harmonizer_dir = f"{self.base_dir}/harmonizer"
            # Use the existing embedding cache in data directory
            harmonizer_cache = f"{self.base_dir}/embedding_cache.pkl"
            scheduler_dir = f"{self.base_dir}/scheduler"
            logbooks_dir = f"{self.base_dir}/logbooks"
            features_dir = f"{self.base_dir}/features"
            rpg_dir = f"{self.base_dir}/rpg"
            routines_dir = f"{self.base_dir}/routines"
            tracking_dir = f"{self.base_dir}/tracking"
            graph_dir = f"{self.base_dir}/graph"
            planner_dir = f"{self.base_dir}/planner"
            orchestrator_dir = f"{self.base_dir}/orchestrator"
            profiles_dir = f"{self.base_dir}/profiles"

            # Create all directories if using base_dir approach
            self.create_directory_structure()

        # Initialize app's OpenAI client (with quirks handling)
        self.openai_client = OpenAIClient()

        # Core components with configured directories
        self.logger = UnifiedLogger(data_dir=tracking_dir)
        self.graph = GraphStore(data_dir=graph_dir)

        # Use configured profiles directory
        self.profile = TemporalPersonalProfile(data_dir=profiles_dir)

        # Initialize label generator
        self.label_generator = LabelGenerator(self.openai_client)

        # Initialize the EnhancedLabelHarmonizer
        # No longer needs harmonization_report.json - works directly with two-tier files
        self.harmonizer = EnhancedLabelHarmonizer(
            harmonization_report_path=f"{harmonizer_dir}/harmonization_report.json",  # Path for compatibility only
            similarity_threshold=0.80,  # For specific groups (>0.8)
            min_semantic_distance=0.5,  # For general groups (>0.5)
            use_real_embeddings=bool(os.getenv('OPENAI_API_KEY')),  # Use embeddings with smart filtering
            embedding_cache_file=harmonizer_cache
        )

        # Initialize harmonizer with seed labels if needed
        self._initialize_harmonizer()

        # Dynamic LogBook System with configured path
        self.logbook = DynamicLogBook(
            base_dir=logbooks_dir,
            openai_client=self.openai_client
        )

        # Detailed Activity Logger (4o talker, 3.5-turbo notetaker)
        self.activity_logger = DetailedActivityLogger(
            logbook=self.logbook,
            logger=self.logger,
            openai_client=self.openai_client
        )

        # LogBook Assistant (can retrieve logs for context)
        self.logbook_assistant = LogBookAssistant(
            logbook=self.logbook,
            openai_client=self.openai_client
        )

        # DUAL SCHEDULER SETUP with configured paths
        # AdaptiveScheduler for pattern analysis, daily scheduling, and pomodoros
        self.adaptive_scheduler = AdaptiveScheduler(
            logger=self.logger,
            graph_store=self.graph,
            profile=self.profile,
            data_dir=scheduler_dir
        )

        # Context-Aware Scheduler for message processing and check-ins
        self.context_scheduler = ContextAwareScheduler(
            logger=self.logger,
            graph_store=self.graph,
            openai_client=self.openai_client,
            labels_dir=labels_dir,  # Uses configured labels dir
            chunks_dir=chunks_dir,  # Uses configured chunks dir
            sandbox_mode=use_sandbox,
            sandbox_name=sandbox_name
        )

        # Default scheduler points to context_scheduler for backward compatibility
        self.scheduler = self.context_scheduler

        # Feature Extractor with configured path
        self.feature_extractor = EnhancedFeatureExtractor(
            logger=self.logger,
            graph_store=self.graph,
            data_dir=features_dir
        )

        # RPG System with configured path
        self.rpg = RPGSystem(
            logger=self.logger,
            graph_store=self.graph,
            data_dir=rpg_dir
        )

        # Routine Builder with configured path
        self.routine_builder = RoutineBuilder(
            logger=self.logger,
            graph_store=self.graph,
            openai_client=self.openai_client,
            data_dir=routines_dir
        )

        # Store paths for later use
        self.paths = {
            'chunks': chunks_dir,
            'labels': labels_dir,
            'harmonizer': harmonizer_dir,
            'orchestrator': orchestrator_dir,
            'planner': planner_dir,
            'logbooks': logbooks_dir,
        }

        # Session management
        self.active_check_in = None
        self.active_detailed_session = None
        self.active_routine = None

        # Tracking for harmonization
        self.labels_created_count = 0
        self.last_harmonization_count = 0

        self.context_matcher = ContextSimilarityMatcher(
            labels_dir=labels_dir,
            chunks_dir=chunks_dir,
            harmonization_dir=harmonizer_dir,
            openai_client=self.openai_client,
            harmonizer=self.harmonizer
        )

        print(f"✅ Complete Integrated System initialized")
        if use_config:
            print(f"   Using centralized configuration")
            print(f"   Data root: C:/BiggerBrother/data")
            print(f"   Labels: {labels_dir}")
            print(f"   Chunks: {chunks_dir}")
            print(f"   Harmonizer: {harmonizer_dir}")
            print(f"   Profiles: {profiles_dir}")
        else:
            print(f"   Base directory: {self.base_dir}")

        print(f"   LogBook categories: {len(self.logbook.categories)}")
        print(f"   Using Two-Tier Harmonization: Yes")
        print(f"   Dual Scheduler Support: Enabled")
        print(f"   Embeddings: Enabled with smart filtering")
        print(f"   Thresholds: General={self.harmonizer.general_threshold}, Specific={self.harmonizer.specific_threshold}")
        print(f"   Embedding cache: {harmonizer_cache}")

        # Show harmonizer state
        if hasattr(self.harmonizer, 'embedding_cache'):
            print(f"   Cached embeddings: {len(self.harmonizer.embedding_cache)}")


        # ---- NEW: Scheduler Bridge (notes → suggestions → crystallize → email)
        self.schedule_bridge = ScheduleBridge(
            planner_dir=planner_dir,
            openai_client=self.openai_client
        )

        # ---- NEW: make sure 'sessions' exists for fallback chat logging
        try:
            self._ensure_log_category(
                "sessions",
                description="Freeform chat sessions (auto-created)",
                fields={"summary": "string", "tokens": "int"}
            )
        except Exception as e:
            print(f"[warn] could not ensure 'sessions' category: {e}")


        # ---- Chat-mode defaults: keep sessions open, no auto-timeout
        self.mode = "chat"
        self.keepalive = True
        try:
            # Soft-configure schedulers if they expose knobs; ignore if not present.
            if hasattr(self.context_scheduler, "configure"):
                self.context_scheduler.configure(mode="chat", session_timeout=None, idle_autoclose=False)
            elif hasattr(self.context_scheduler, "set_mode"):
                self.context_scheduler.set_mode("chat")
            if hasattr(self.adaptive_scheduler, "configure"):
                self.adaptive_scheduler.configure(mode="chat", session_timeout=None)
        except Exception as e:
            print(f"[warn] scheduler chat-mode config skipped: {e}")

    # --- Compose a conversational reply for chat mode -----------------------
    def _compose_chat_reply(self, message: str, similar_messages: List[str], harmonized: Dict, max_ctx: int = 3) -> str:
        """
        Produce a user-facing reply that actually *uses* context and harmonized labels.
        Never asks to end the session; designed for free-form chat.
        """
        # Summarize labels succinctly (specific groups first)
        spec_topics = [l.get('specific_group', l.get('label')) for l in harmonized.get('topic', [])][:3]
        tones       = [l.get('label') for l in harmonized.get('tone', [])][:2]
        intents     = [l.get('label') for l in harmonized.get('intent', [])][:2]

        ctx_snips = []
        for s in similar_messages:
            if not isinstance(s, str):
                continue
            txt = s.strip()
            if not txt:
                continue
            # keep snippets short, plain
            ctx_snips.append(txt)

        sys_prompt = (
            "You are a collaborator and friend. "
            "Use the user's current message, and the context to compose a personalized message. "
            "Be concrete and grounded in the context; do not invent details not implied by them. "
            "DO NOT try to end the conversation or suggest ending; "
            "keep the tone open-ended."
        )

        ctx_blob = json.dumps({
            "topics_specific": spec_topics,
            "tones": tones,
            "intents": intents,
            "context_snippets": ctx_snips
        }, ensure_ascii=False)

        try:
            reply = self.openai_client.chat(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": f"CURRENT MESSAGE:\n{message}\n\nLABELED CONTEXT:\n{ctx_snips}"}
                ],
                model="gpt-5"
            )
            return reply.strip()
        except Exception as e:
            # Fall back to a minimal, non-generic template
            # (still contains detected topics so it isn't a blank platitude)
            hint = ", ".join([t for t in spec_topics if t]) or "your themes"
            return f"I hear you. Based on {hint}, I’ve pulled a few relevant notes from your history. Want to go deeper on one of these or take an actionable next step?"

    # ==================== NEW TWO-TIER CONTEXT METHODS ====================

    def get_similar_messages_for_context(self, message: str, current_labels: Dict[str, List[Dict]], limit: int = 100) -> List[str]:
        # Delegate to the new matcher
        context_messages, metadata = self.context_matcher.find_similar_messages(
            message=message,  # We already have labels, could refactor this
            min_chars_recent=50000,
            min_chars_long_term=150000
        )
        return [msg["content"] for msg in context_messages[:limit]]

    def select_context_categories_by_labels(self, harmonized_labels: Dict) -> Tuple[List[str], int, int]:
        """
        Select logbook categories based on harmonized labels.
        Returns (categories, days_back, max_entries)
        """
        # Map label groups to logbook categories
        category_scores = defaultdict(float)

        # Mapping of keywords to categories
        keyword_mapping = {
            'medication': 'medications',
            'medicine': 'medications',
            'pill': 'medications',
            'dose': 'medications',
            'food': 'meals',
            'eat': 'meals',
            'meal': 'meals',
            'breakfast': 'meals',
            'lunch': 'meals',
            'dinner': 'meals',
            'exercise': 'exercise',
            'workout': 'exercise',
            'run': 'exercise',
            'walk': 'exercise',
            'mood': 'mood',
            'feeling': 'mood',
            'emotion': 'mood',
            'happy': 'mood',
            'sad': 'mood',
            'sleep': 'sleep',
            'rest': 'sleep',
            'tired': 'sleep',
            'awake': 'sleep',
            'work': 'productivity',
            'task': 'productivity',
            'project': 'productivity',
            'meeting': 'productivity',
            'symptom': 'symptoms',
            'pain': 'symptoms',
            'headache': 'symptoms',
            'sick': 'symptoms'
        }

        # Score categories based on label probabilities
        for label_item in harmonized_labels.get('topic', []):
            label = label_item.get('label', '').lower()
            prob = label_item.get('p', 1.0)

            for keyword, category in keyword_mapping.items():
                if keyword in label:
                    category_scores[category] += prob

        # Sort categories by score
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        categories = [cat for cat, _ in sorted_categories[:3]]  # Top 3 categories

        # Determine depth based on similarity index
        total_similarity = 0.0
        for category in ['topic', 'tone', 'intent']:
            labels = harmonized_labels.get(category, [])
            if labels:
                general = set(l.get('general_group', l['label']) for l in labels)
                specific = set(l.get('specific_group', l['label']) for l in labels)
                if general:
                    total_similarity += len(specific) / len(general)

        avg_similarity = total_similarity / 3 if total_similarity > 0 else 0.5

        if avg_similarity > 0.7:
            # High similarity - focused context
            return categories, 3650, 36500
        elif avg_similarity < 0.3:
            # Low similarity - broad context
            return categories, 3650, 36500
        else:
            # Medium similarity
            return categories, 3650, 36500

    # ==================== UPDATED MAIN METHODS ====================

    def process_message_with_context(self, message: str) -> Dict:
        """
        Process a message with context awareness using probability-weighted similarity.
        Uses harmonized labels with probabilities to find similar messages for context.
        """
        # Generate and harmonize labels FIRST to determine context needs
        raw_labels = self._generate_raw_labels(message)
        harmonized = self.harmonizer.process_label_set(raw_labels, message_context=message)

        # Find similar messages using probability-weighted similarity
        if hasattr(self, 'context_matcher'):
            context_messages, metadata = self.context_matcher.find_similar_messages(
                message,
                min_chars_recent=50000,
                min_chars_long_term=100000
            )

            # Load messages up to character limit, not message count
            similar_messages = []
            char_count = 0
            target_chars = 150000  # Adjust as needed

            for msg in context_messages:
                msg_content = msg.get("content", "")
                char_count += len(msg_content)
                similar_messages.append(msg_content)
                if char_count >= target_chars:
                    break

            print(f"   Found {len(similar_messages)} similar messages ({char_count} chars) for context")
        else:
            similar_messages = []
            print("   No context matcher initialized")

        # Select categories and depth based on harmonized labels
        categories_needed, days_back, max_entries = self.select_context_categories_by_labels(harmonized)

        # Load targeted context
        log_context = {}
        if categories_needed:
            print(f"📚 Loading context from: {categories_needed} (depth: {days_back}d, {max_entries} entries)")
            for category in categories_needed:
                if category in self.logbook.categories:
                    entries = self.logbook.load_category_context(
                        category,
                        days_back=days_back,
                        max_entries=max_entries
                    )
                    if entries:
                        log_context[category] = entries

        # Add similar messages to context (debug print)
        if similar_messages:
            print(f"   Found {len(similar_messages)} similar messages for context")

        # ---- NEW: Capture schedule-worthy note opportunistically
        schedule_note = None
        try:
            intents = [l.get("label","").lower() for l in harmonized.get("intent", [])]
            text_l = message.lower()
            _triggers = {"plan", "planning", "plan_next_steps", "schedule", "scheduling", "appointment"}
            if any(t in intents for t in _triggers) or any(w in text_l for w in ("schedule", "plan", "appointment", "tomorrow", "today")):
                schedule_note = self.schedule_bridge.capture_note(message, harmonized, source="chat")
        except Exception as e:
            print(f"[warn] schedule note capture failed: {e}")
        # Compose a user-facing chat reply (chat mode)
        assistant_text = self._compose_chat_reply(message, similar_messages, harmonized)

        # Process based on active session type
        if self.active_check_in:
            # Process through check-in
            scheduler_response = self.scheduler.process_message(message)

            response = {
                "response": scheduler_response.get("response", ""),
                "extracted": [],
                "logged": []
            }

            # Extract to log books with harmonized labels and context
            extracted_data = self._extract_to_logbooks(message, self.active_check_in["session_id"], harmonized)
            response["extracted"] = extracted_data.get("extracted", [])
            response["logged"] = extracted_data.get("logged", [])
            response["label_insights"] = extracted_data.get("label_insights", {})

            # Add scheduler-specific fields
            response["labels"] = scheduler_response.get("labels", [])
            response["context_size"] = scheduler_response.get("context_size", 0)
            response["similar_messages_count"] = len(similar_messages)
            response["similar_messages"] = similar_messages

        elif self.active_detailed_session:
            # Process through detailed activity logger
            response = self.activity_logger.process_activity_description(message)

        else:
            # Quick log with context (chat mode)
            response = self._quick_log_with_context(message, log_context, harmonized)
            # If extractor produced nothing, ensure we still log a minimal session entry
            try:
                logged_any = bool(response.get("logged"))
            except Exception:
                logged_any = False
            if not logged_any:
                try:
                    if hasattr(self.logbook, "log_entry"):
                        self.logbook.log_entry(
                            category_name="sessions",
                            data={"summary": "chat_message", "tokens": len(message.split())},
                            raw_text=message,
                            extracted_by="chat_fallback"
                        )
                        response.setdefault("logged", []).append("sessions")
                except Exception as e:
                    print(f"[warn] fallback session log failed: {e}")

            # Opportunistic passive scheduling: let scheduler sniff the message for tasks, but never close chat.
            try:
                if hasattr(self.scheduler, "process_message") and os.getenv("BB_CHAT_SCHEDULER", "1") == "1":
                    sched_out = self.scheduler.process_message(message)
                    # attach a light summary if it exists
                    if isinstance(sched_out, dict):
                        response["scheduler"] = {k: v for k, v in sched_out.items() if k in ("tasks", "labels", "context_size") or isinstance(v, (str, int, float, list, dict))}
            except Exception as e:
                print(f"[warn] scheduler passive call skipped: {e}")

        # Save harmonized labels to disk
        if harmonized:
            labels_dir = f"{self.base_dir}/labels"
            os.makedirs(labels_dir, exist_ok=True)

            msg_hash = hashlib.md5(message.encode()).hexdigest()[:8]
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            label_file = f"{labels_dir}/{timestamp}_{msg_hash}.json"

            with open(label_file, 'w') as f:
                # Try to fetch the gid for *this* message from the matcher's just-created label file
                cur_gid = None
                try:
                    if isinstance(metadata, dict) and metadata.get("label_file"):
                        lf = Path(self.paths.get("labels", labels_dir)) / metadata["label_file"]
                        with open(lf, "r", encoding="utf-8") as _lf:
                            cur_gid = (json.load(_lf) or {}).get("gid")
                except Exception:
                    pass

                payload = {
                    "gid": cur_gid,
                    "raw_labels": raw_labels,
                    "harmonized": harmonized,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "similar_messages": similar_messages[:30000]
                }
                if isinstance(metadata, dict):
                    if metadata.get("similar_message_gids"):
                        payload["similar_message_gids"] = metadata["similar_message_gids"]
                    if metadata.get("similarity_debug"):
                        payload["similarity_debug"] = metadata["similarity_debug"][:200]
                json.dump(payload, f, indent=2)

            # Count labels created
            total_labels = sum(len(items) for items in harmonized.values())
            self.labels_created_count += total_labels

            if self.needs_harmonization():
                print(f"⚠️ Harmonization recommended: {self.labels_created_count - self.last_harmonization_count} new labels")
                # By default, skip the expensive full rebuild; rely on write-through during matching.
                if os.getenv("BB_AUTO_REHARMONIZE", "0") == "1":
                    if self.labels_created_count - self.last_harmonization_count >= 1:
                        print("🔄 Updating two-tier harmonization groups...")
                        for category in ['topic', 'tone', 'intent']:
                            self.harmonizer._update_harmonization_tier(category)
                        self.harmonizer.save_harmonization_tiers()
                        self.reset_harmonization_counter()
                        print("   ✅ Two-tier groups updated")
                else:
                    # Skipped full rebuild. Incremental write-through keeps harmonization files fresh.
                    pass

        # Extract features
        features = self.feature_extractor.extract_message_features(
            message,
            datetime.now(timezone.utc)
        )

        response["response"] = assistant_text  # <-- ensure chat has an actual reply
        response["keep_alive"] = True  # <-- front-end can use this to avoid auto-closing chat
        response["mode"] = "chat"
        # Keep chat sessions open; some UIs treat missing field as 'end'
        response.setdefault("response", "")
        response["keep_alive"] = True
        response["mode"] = "chat"

        response["features"] = {
            "vocabulary_richness": features.vocabulary_richness,
            "rare_words": features.rare_word_count,
            "engagement": features.word_count > 20
        }

        # Add context info to response
        response["context_analysis"] = {
            "similar_messages": len(similar_messages),
            "categories_loaded": categories_needed,
            "context_depth": f"{days_back} days, {max_entries} entries"
        }

        response["labels_generated"] = self.labels_created_count - self.last_harmonization_count

        #print(f"{response}")
        if log_context:
            response["log_context_loaded"] = list(log_context.keys())

        # Surface schedule-bridge note id if we captured one
        if schedule_note:
            response["schedule_note_id"] = schedule_note["id"]

        # Let the scheduler bridge send any reminders that are now due
        try:
            self.schedule_bridge.reminders.tick()
        except Exception:
            pass

        return response

    def _extract_to_logbooks(self, message: str, session_id: str, harmonized: Dict = None) -> Dict:
        """
        Extract activities from a message and log to appropriate books.
        Uses harmonized labels with two-tier information for better categorization.
        """
        extracted_data = {"extracted": [], "logged": [], "label_insights": {}}

        # If harmonized not provided, generate it
        if not harmonized:
            raw_labels = self._generate_raw_labels(message)
            harmonized = self.harmonizer.process_label_set(raw_labels, message_context=message)

        # Store insights about labels including tier information
        for category, labels in harmonized.items():
            # Count unique concepts
            unique_count = sum(1 for l in labels if l.get('original') == l.get('label'))

            # NEW: Count tier alignments
            general_groups = set(l.get('general_group', l['label']) for l in labels)
            specific_groups = set(l.get('specific_group', l['label']) for l in labels)

            extracted_data['label_insights'][category] = {
                'total': len(labels),
                'unique_concepts': unique_count,
                'general_groups': len(general_groups),
                'specific_groups': len(specific_groups),
                'similarity_index': len(specific_groups) / len(general_groups) if general_groups else 1.0
            }

        # Build extraction prompt using both tiers
        specific_topics = [l.get('specific_group', l['label']) for l in harmonized.get('topic', [])]
        general_topics = [l.get('general_group', l['label']) for l in harmonized.get('topic', [])]

        extraction_prompt = f"""Extract structured activities from this message.
        
        Specific topics: {specific_topics[:3]}
        General themes: {list(set(general_topics))[:3]}
        Detected tone: {[l['label'] for l in harmonized.get('tone', [])[:2]]}
        Detected intent: {[l['label'] for l in harmonized.get('intent', [])[:2]]}
        
        Categories: medications, meals, exercise, mood, sleep, productivity, symptoms
        
        Return JSON array of activities:
        [
            {{
                "category": "category_name",
                "data": {{relevant fields}},
                "confidence": 0.0-1.0,
                "related_topics": []
            }}
        ]
        """

        try:
            response = self.openai_client.chat(
                messages=[
                    {"role": "system", "content": extraction_prompt},
                    {"role": "user", "content": message}
                ],
                model="gpt-5-nano"
            )

            # Parse response
            content = response.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]

            activities = json.loads(content) if content.startswith('[') else [json.loads(content)]
            extracted_data["extracted"] = activities

            # Log each activity with harmonized context
            for activity in activities:
                category = activity.get("category")
                if category and category in self.logbook.categories:
                    try:
                        # Add harmonized label context with tier info
                        activity['data']['label_context'] = {
                            'specific_topics': specific_topics[:2],
                            'general_themes': list(set(general_topics))[:2],
                            'tone': harmonized.get('tone', [{}])[0].get('label') if harmonized.get('tone') else None
                        }

                        self.logbook.log_entry(
                            category_name=category,
                            data=activity.get("data", {}),
                            raw_text=message,
                            extracted_by="check_in",
                            confidence=activity.get("confidence", 0.8),
                            session_id=session_id
                        )
                        extracted_data["logged"].append(category)
                    except Exception as e:
                        print(f"Error logging to {category}: {e}")

        except Exception as e:
            print(f"Extraction error: {e}")

        return extracted_data

    def _quick_log_with_context(self, message: str, context: Dict, harmonized: Dict = None) -> Dict:
        """
        Quick log with relevant context loaded.
        Uses harmonized labels to improve extraction.
        """
        # Build context prompt
        context_prompt = ""
        if context:
            context_prompt = f"\nRelevant log context:\n{json.dumps(context, indent=2)}"

        # Add harmonization info if available
        if harmonized:
            specific_topics = [l.get('specific_group', l['label']) for l in harmonized.get('topic', [])]
            general_topics = [l.get('general_group', l['label']) for l in harmonized.get('topic', [])]

            context_prompt += f"\nDetected topics (specific): {specific_topics[:3]}"
            context_prompt += f"\nDetected themes (general): {list(set(general_topics))[:3]}"

        # Extract with context
        prompt = f"""Extract structured data from this message.
        {context_prompt}
        
        Message: {message}
        
        Extract and categorize activities."""

        try:
            response = self.openai_client.chat(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": message}
                ],
                model="gpt-5-nano"
            )

            # Parse and log
            extracted = json.loads(response) if response.startswith('[') or response.startswith('{') else {"entries": []}
            logged = []

            for item in extracted.get("entries", []):
                category = item.get("category")
                # NEW: auto-create category if missing
                if category and category not in self.logbook.categories:
                    try:
                        self._ensure_log_category(category, description=f"Auto-created by extractor", fields={})
                    except Exception as e:
                        print(f"[warn] could not create log category '{category}': {e}")
                if category in self.logbook.categories:
                    self.logbook.log_entry(
                        category_name=category,
                        data=item.get("data", {}),
                        raw_text=message,
                        extracted_by="quick_log"
                    )
                    logged.append(category)

            # Minimal fallback: always record that we saw a chat message
            if not logged:
                try:
                    self._ensure_log_category("sessions", description="Freeform chat sessions (auto-created)",
                                              fields={"summary": "string", "tokens": "int"})
                    self.logbook.log_entry(
                        category_name="sessions",
                        data={"summary": "chat_message", "tokens": len(message.split())},
                        raw_text=message,
                        extracted_by="chat_fallback"
                    )
                    logged.append("sessions")
                except Exception as e:
                    print(f"[warn] fallback session log failed: {e}")

            return {
                "logged": logged,
                "extracted": extracted.get("entries", []),
                "context_used": list(context.keys()) if context else []
            }

        except Exception as e:
            print(f"Quick log error: {e}")
            return {"error": str(e)}


    # --------- NEW: public helpers to use the Scheduler Bridge ----------
    def review_schedule_notes_for_day(self, date=None) -> Dict:
        """Return schedule suggestions from captured notes for the given day (local ET by default)."""
        return self.schedule_bridge.get_suggestions_for_day(date)

    def crystallize_schedule_for_day(self, tasks: List[Dict], date=None,
                                     tz: str = "America/New_York", send_emails: bool = True) -> Dict:
        """Finalize a day's schedule and queue email reminders at task start."""
        return self.schedule_bridge.crystallize_schedule(tasks, date=date, tz=tz, send_emails=send_emails)

    def _ensure_log_category(self, name: str, description: str = "", fields: Optional[Dict[str, Any]] = None):
        """Create a logbook category if missing (tries logbook API, then falls back to filesystem)."""
        if hasattr(self.logbook, "categories") and name in self.logbook.categories:
            return
        fields = fields or {}
        # Try DynamicLogBook's own APIs if present
        for method in ("create_category", "ensure_category", "register_category"):
            if hasattr(self.logbook, method):
                try:
                    getattr(self.logbook, method)(name=name, description=description, fields=fields)
                    return
                except Exception as e:
                    print(f"[warn] logbook.{method} failed for '{name}': {e}")
        # Filesystem fallback: create directory + seed files, then ask logbook to reload
        cat_dir = Path(self.paths["logbooks"]) / name
        cat_dir.mkdir(parents=True, exist_ok=True)
        readme = (cat_dir / "README.md")
        if not readme.exists():
            readme.write_text(f"# {name}\n\n{description}\n", encoding="utf-8")
        (cat_dir / f"{name}.jsonl").touch(exist_ok=True)
        # Try to reload categories if supported
        if hasattr(self.logbook, "reload_categories"):
            try:
                self.logbook.reload_categories()
            except Exception:
                pass

    # Convenience methods for AdaptiveScheduler functionality
    def analyze_behavioral_patterns(self, days: int = 30):
        """Analyze patterns using adaptive scheduler."""
        return self.adaptive_scheduler.analyze_patterns(days=days)

    def generate_daily_schedule(self, date=None):
        """Generate daily schedule using adaptive scheduler."""
        if date is None:
            date = datetime.now(timezone.utc)
        return self.adaptive_scheduler.generate_daily_schedule(date)

    def schedule_pomodoro(self, task: str, start_time=None, duration=25, break_duration=5):
        """Schedule a pomodoro session using adaptive scheduler."""
        return self.adaptive_scheduler.schedule_pomodoro(
            task, start_time, duration, break_duration
        )

    def _initialize_harmonizer(self):
        """
        Initialize the harmonizer from two-tier files if they exist.
        No longer uses harmonization_report.json.
        """
        # Check if two-tier files exist
        general_file = Path(self.base_dir) / "data" / "harmonization_general.json"
        specific_file = Path(self.base_dir) / "data" / "harmonization_specific.json"

        files_exist = general_file.exists() or specific_file.exists()

        if files_exist:
            print(f"\n✅ Harmonizer initialized from two-tier files")

            # Report what was loaded
            total_labels = sum(len(self.harmonizer.label_frequencies[cat]) for cat in ['topic', 'tone', 'intent'])
            if total_labels > 0:
                print(f"   Loaded {total_labels:,} unique labels")

            # Report two-tier statistics
            print("\n📊 Two-Tier Harmonization Status:")
            for category in ['topic', 'tone', 'intent']:
                general_count = len(self.harmonizer.general_groups[category])
                specific_count = len(self.harmonizer.specific_groups[category])

                if general_count > 0 or specific_count > 0:
                    print(f"\n   {category}:")
                    print(f"      General groups (>0.5): {general_count}")
                    print(f"      Specific groups (>0.8): {specific_count}")

                    if general_count > 0 and specific_count > 0:
                        ratio = specific_count / general_count
                        print(f"      Specificity ratio: {ratio:.2f}")

            # Report embedding cache status
            if self.harmonizer.embedding_cache:
                print(f"\n   Embedding cache: {len(self.harmonizer.embedding_cache)} cached embeddings")

            print("\n   System uses smart filtering to minimize API calls:")
            print("     - String matching first (no API)")
            print("     - Groups similar strings together")
            print("     - Only uses embeddings for ambiguous cases (0.3 < similarity < 0.85)")
            print("     - Preserves label probabilities for context weighting")
        else:
            print("\n📝 No existing two-tier harmonization files found")
            print("   The harmonizer will build groups as labels are processed")
            print("   To build initial harmonization:")
            print("     1. Process some messages to generate labels")
            print("     2. Or run enhanced_smart_label_importer.py if you have existing labels")
            print("\n   Files that will be created:")
            print(f"     - {general_file}")
            print(f"     - {specific_file}")

    def needs_harmonization(self, threshold: int = 1) -> bool:
        """Check if harmonizer should be run based on new labels created."""
        return (self.labels_created_count - self.last_harmonization_count) >= threshold

    def reset_harmonization_counter(self):
        """Call after running harmonizer."""
        self.last_harmonization_count = self.labels_created_count

    def rebuild_two_tier_groups(self) -> Dict:
        """
        Manually rebuild the two-tier harmonization groups.
        Useful after importing many new labels or adjusting thresholds.
        """
        print("\n🔄 Rebuilding two-tier harmonization groups...")

        stats = {
            'before': {},
            'after': {}
        }

        # Record before state
        for category in ['topic', 'tone', 'intent']:
            stats['before'][category] = {
                'general': len(self.harmonizer.general_groups[category]),
                'specific': len(self.harmonizer.specific_groups[category])
            }

        # Rebuild groups
        for category in ['topic', 'tone', 'intent']:
            if self.harmonizer.label_frequencies[category]:
                self.harmonizer._update_harmonization_tier(category)

        # Save the updated groups
        self.harmonizer.save_harmonization_tiers()
        self.harmonizer.update_harmonization_report()

        # Record after state
        for category in ['topic', 'tone', 'intent']:
            stats['after'][category] = {
                'general': len(self.harmonizer.general_groups[category]),
                'specific': len(self.harmonizer.specific_groups[category])
            }

        # Reset harmonization counter
        self.reset_harmonization_counter()

        print("✅ Two-tier groups rebuilt successfully!")

        # Show changes
        for category in ['topic', 'tone', 'intent']:
            before = stats['before'][category]
            after = stats['after'][category]

            if before != after:
                print(f"\n   {category}:")
                print(f"      General: {before['general']} → {after['general']}")
                print(f"      Specific: {before['specific']} → {after['specific']}")

        return stats

    def get_harmonization_insights(self) -> Dict:
        """Get insights about label harmonization and grouping."""
        insights = {
            'categories': {},
            'total_labels': 0,
            'total_unique': 0,
            'needs_harmonization': self.needs_harmonization(),
            'embeddings_cached': len(self.harmonizer.embedding_cache)
        }

        for category in ['topic', 'tone', 'intent']:
            # Get statistics from harmonizer
            freq_count = len(self.harmonizer.label_frequencies[category])
            general_count = len(self.harmonizer.general_groups[category])
            specific_count = len(self.harmonizer.specific_groups[category])

            # Count unique concepts (those not mapped to other labels)
            unique_concepts = 0
            for label in self.harmonizer.label_frequencies[category]:
                if label not in self.harmonizer.label_mappings[category]:
                    unique_concepts += 1

            # Prepare category stats
            category_stats = {
                'original_count': freq_count,
                'canonical_count': freq_count - len(self.harmonizer.label_mappings[category]),
                'unique_concepts': unique_concepts,
                'two_tier': {
                    'general_groups': general_count,
                    'specific_groups': specific_count,
                    'specificity_ratio': specific_count / general_count if general_count > 0 else 0
                }
            }

            insights['categories'][category] = category_stats
            insights['total_labels'] += category_stats['canonical_count']
            insights['total_unique'] += unique_concepts

        return insights

    def start_check_in_with_logs(
            self,
            check_type: Optional[CheckInType] = None,
            include_summary: bool = True
    ) -> Dict[str, Any]:
        """
        Start a check-in session with recent logs as context.
        CRITICAL FIX: Now properly sets active_check_in!

        Args:
            check_type: Type of check-in (auto-detected if None)
            include_summary: Whether to include a summary of recent activity

        Returns:
            Session information
        """
        # Build context from recent logs
        context = {}

        if include_summary:
            # Get recent activity summary
            last_hour = datetime.now(timezone.utc) - timedelta(hours=1)
            recent_logs = self.graph.search_nodes(
                node_type="log",
                created_after=last_hour.isoformat() if hasattr(self.graph, 'search_nodes') else None,
                limit=20
            ) if hasattr(self.graph, 'search_nodes') else []

            if recent_logs:
                categories = set()
                for log in recent_logs:
                    if 'attrs' in log:
                        cat = log['attrs'].get('category', 'unknown')
                        categories.add(cat)

                context['recent_activity'] = {
                    'log_count': len(recent_logs),
                    'categories': list(categories),
                    'time_range': 'last_hour'
                }

        # Let the scheduler auto-detect type if None (uses context_scheduler)
        session = self.scheduler.start_check_in(check_type, context)

        # CRITICAL FIX: Set active_check_in so process_message knows we're in a session!
        self.active_check_in = session

        return session

    def _generate_raw_labels(self, message: str) -> Dict:
        """Generate raw labels from message for harmonization."""
        try:
            prompt = """Analyze this message and extract labels.
            Return JSON with structure:
            {
                "topic": [{"label": "...", "p": 0.0-1.0}],
                "tone": [{"label": "...", "p": 0.0-1.0}],
                "intent": [{"label": "...", "p": 0.0-1.0}]
            }"""

            response = self.openai_client.chat(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": message}
                ],
                model="gpt-5-nano"
            )

            content = response.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]

            return json.loads(content)
        except Exception as e:
            print(f"Error generating labels: {e}")
            return {"topic": [], "tone": [], "intent": []}

    def start_detailed_activity_logging(self) -> Dict:
        """
        Start detailed activity logging with GPT-4o conversationalist
        and gpt-5-nano extractor.
        """
        session = self.activity_logger.start_detailed_logging()
        self.active_detailed_session = session

        return session

    def start_routine_with_logging(self, routine_name: str) -> Dict:
        """
        Start a routine that logs each step to appropriate log books.
        """
        result = self.routine_builder.start_routine(routine_name)
        self.active_routine = routine_name

        # Log routine start
        self.logbook.log_entry(
            category_name="productivity",
            data={
                "tasks_completed": [],
                "focus_score": 0,
                "pomodoros": 0,
                "blockers": [],
                "achievements": [f"Started {routine_name}"]
            },
            extracted_by="system"
        )

        return result

    def complete_routine_step(self, step_name: str, notes: str = "") -> Dict:
        """
        Complete a routine step with logging.
        """
        result = self.routine_builder.complete_step(step_name, notes)

        # Determine log category based on step
        routine = self.routine_builder.routines.get(self.active_routine)
        if routine:
            for step in routine.steps:
                if step.name == step_name:
                    # Log to appropriate category
                    if step.category == "medication":
                        self.logbook.log_entry(
                            category_name="medications",
                            data={
                                "medication": step.name,
                                "dose": "as prescribed",
                                "time": datetime.now(timezone.utc).strftime("%H:%M"),
                                "with_food": False,
                                "notes": notes
                            },
                            extracted_by="routine"
                        )
                    elif step.category == "hygiene":
                        # Could create a hygiene log book if needed
                        pass

                    break

        return result

    def query_with_logs(self, query: str) -> str:
        """
        Answer a query using log book context.
        """
        return self.logbook_assistant.answer_with_logs(query)

    def propose_new_log_category(self, conversation_text: str) -> Optional[Dict]:
        """
        Have gpt-5-nano propose a new log category based on conversation.
        """
        proposed = self.logbook.propose_category(conversation_text)

        if proposed:
            # Award XP for creating a new tracking category
            self.rpg.process_activity(
                "organization",
                1,
                {"action": "created_log_category", "category": proposed.name}
            )

            return proposed.to_dict()

        return None

    def get_daily_summary_with_logs(self) -> Dict:
        """
        Get comprehensive daily summary including all log books with harmonization stats.
        """
        today = datetime.now(timezone.utc).date()

        # Get log book summaries
        log_summaries = {}
        for category_name, category in self.logbook.categories.items():
            entries = self.logbook.load_category_context(
                category_name,
                days_back=365,
                max_entries=500000
            )

            # Filter to today
            today_entries = [
                e for e in entries
                if ensure_timezone_aware(datetime.fromisoformat(e["timestamp"])).date() == today
            ]

            if today_entries:
                log_summaries[category_name] = {
                    "count": len(today_entries),
                    "entries": today_entries[:5]  # First 5 for summary
                }

        # Get activity features
        activity_features = self.feature_extractor.extract_activity_features(
            ensure_timezone_aware(datetime.combine(today, datetime.min.time())).replace(tzinfo=timezone.utc),
            datetime.now(timezone.utc)
        )

        # Get RPG progress
        rpg_summary = self.rpg.get_character_summary()

        # Get harmonization insights
        harmonization = self.get_harmonization_insights()

        return {
            "date": today.isoformat(),
            "log_books": log_summaries,
            "log_categories_active": len(log_summaries),
            "total_logs_today": sum(s["count"] for s in log_summaries.values()),
            "activity_metrics": {
                "tasks_completed": activity_features.tasks_completed,
                "focus_sessions": activity_features.focus_sessions,
                "exercise_minutes": activity_features.exercise_minutes,
                "productivity_score": self.feature_extractor._calculate_productivity_score(activity_features),
                "wellness_score": self.feature_extractor._calculate_wellness_score(activity_features)
            },
            "rpg": {
                "level": rpg_summary["character"]["level"],
                "xp_today": rpg_summary["character"]["total_xp"],
                "skills_practiced": len([s for s in rpg_summary["top_skills"] if s[2] > 0])
            },
            "harmonization": {
                "total_groups": harmonization['total_labels'],
                "unique_concepts": harmonization['total_unique'],
                "needs_update": harmonization['needs_harmonization'],
                "two_tier_summary": {
                    "general_groups": sum(c['two_tier']['general_groups'] for c in harmonization['categories'].values()),
                    "specific_groups": sum(c['two_tier']['specific_groups'] for c in harmonization['categories'].values())
                }
            }
        }

    def create_directory_structure(self):
        """
        Create the complete directory structure for the system.
        """
        directories = [
            f"{self.base_dir}/tracking",      # UnifiedLogger
            f"{self.base_dir}/graph",         # GraphStore
            f"{self.base_dir}/logbooks",      # DynamicLogBook
            f"{self.base_dir}/logbooks/medications",
            f"{self.base_dir}/logbooks/meals",
            f"{self.base_dir}/logbooks/exercise",
            f"{self.base_dir}/logbooks/mood",
            f"{self.base_dir}/logbooks/sleep",
            f"{self.base_dir}/logbooks/productivity",
            f"{self.base_dir}/logbooks/symptoms",
            f"{self.base_dir}/logbooks/sessions",  # Detailed activity sessions
            f"{self.base_dir}/labels",        # Label storage
            f"{self.base_dir}/chunks",        # Message chunks
            f"{self.base_dir}/features",      # Extracted features
            f"{self.base_dir}/rpg",          # RPG data
            f"{self.base_dir}/routines",      # Behavioral routines
            f"{self.base_dir}/planner",       # Daily plans
            f"{self.base_dir}/scheduler",     # AdaptiveScheduler data
            f"{self.base_dir}/harmonizer",    # Harmonizer data
            f"{self.base_dir}/orchestrator",  # Orchestrator data
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        print(f"✅ Created directory structure under {self.base_dir}/")

        # Create README in logbooks
        readme_content = """# LogBooks Directory

Each subdirectory is a separate log category with:
- `{category}.csv` - Structured log entries
- `{category}.jsonl` - Full log entries with metadata
- `daily_YYYYMMDD.json` - Daily summaries
- `README.md` - Category documentation

Categories are created dynamically based on:
1. User requests
2. gpt-5-nano proposals from conversations
3. System defaults

The system uses Two-Tier Harmonization:
- General groups (>0.5 similarity) for broad concepts
- Specific groups (>0.8 similarity) for fine-grained matching
- Groups are used as fast lookup tables for context selection
"""

        with open(f"{self.base_dir}/logbooks/README.md", "w") as f:
            f.write(readme_content)


# Main CLI for the complete system
def main():
    """Interactive CLI for the complete integrated system."""
    import sys

    print("🧠 BiggerBrother - Complete Behavioral Intelligence System")
    print("   with Two-Tier Harmonization and Fast Context Lookups")
    print("=" * 60)

    # Initialize system
    system = CompleteIntegratedSystem(base_dir="data")

    # Show available log categories and harmonization status
    print(f"\n📚 Available Log Categories:")
    for cat in system.logbook.get_categories_for_context():
        print(f"   {cat['name']}: {cat['entry_count']} entries")

    # Show harmonization insights
    insights = system.get_harmonization_insights()
    print(f"\n📏 Label Organization:")
    for category, stats in insights['categories'].items():
        tier_info = stats.get('two_tier', {})
        print(f"   {category}: {tier_info.get('general_groups', 0)} general, {tier_info.get('specific_groups', 0)} specific groups")

    print("\n" + "=" * 60)
    print("Commands:")
    print("  checkin     - Start structured check-in")
    print("  detailed    - Start detailed activity logging (4o + 3.5-turbo)")
    print("  routine     - Start a behavioral routine")
    print("  quick       - Quick log entry")
    print("  query       - Query with log book context")
    print("  propose     - Propose new log category")
    print("  summary     - Daily summary with all logs")
    print("  harmonize   - View harmonization status")
    print("  rebuild     - Rebuild two-tier harmonization groups")
    print("  patterns    - Analyze behavioral patterns")
    print("  schedule    - View today's schedule")
    print("  quit        - Exit")
    print("=" * 60)

    while True:
        try:
            command = input("\n> ").strip().lower()

            if command == "quit":
                break

            elif command == "checkin":
                # Start check-in with log context
                session = system.start_check_in_with_logs()
                print(f"\n{session['greeting']}")

                while system.active_check_in:
                    user_input = input("You: ")
                    if user_input.lower() in ["done", "exit"]:
                        system.active_check_in = None
                        print("✅ Check-in complete")
                        break

                    response = system.process_message_with_context(user_input)
                    print(f"Assistant: {response['response']}")

                    # Show context analysis
                    if response.get("context_analysis"):
                        ctx = response["context_analysis"]
                        print(f"   [Context: {ctx['specific_matches']} specific, {ctx['general_matches']} general matches]")
                        if ctx.get('categories_loaded'):
                            print(f"   [Loaded from: {', '.join(ctx['categories_loaded'])}]")

            elif command == "detailed":
                # Start detailed activity logging
                session = system.start_detailed_activity_logging()
                print(f"\n{session['greeting']}")

                while system.active_detailed_session:
                    user_input = input("You: ")
                    if user_input.lower() in ["done", "exit"]:
                        summary = system.activity_logger.complete_session()
                        system.active_detailed_session = None
                        print(f"✅ Session complete: {summary['activities_extracted']} activities logged")
                        break

                    response = system.process_message_with_context(user_input)
                    print(f"Assistant: {response['response']}")
                    if response.get("logged_to"):
                        print(f"   [Logged to: {', '.join(response['logged_to'])}]")

            elif command == "routine":
                # Show available routines
                routines = system.routine_builder.get_all_routines()
                print("\nAvailable routines:")
                for name, info in routines.items():
                    print(f"  - {name}: {info['steps']} steps, {info['estimated_minutes']:.0f} min")

                routine_name = input("Routine to start: ")
                if routine_name in routines:
                    result = system.start_routine_with_logging(routine_name)
                    print(f"Started {routine_name}")

                    # Simple step completion
                    while True:
                        action = input("Complete step (name) or 'done': ")
                        if action == "done":
                            completion = system.routine_builder.complete_routine()
                            print(f"✅ Routine complete in {completion['result']['duration_minutes']:.1f} minutes")
                            break
                        else:
                            result = system.complete_routine_step(action)
                            if "next_step" in result:
                                print(f"✓ Completed. Next: {result.get('next_step', 'Done!')}")

            elif command == "quick":
                message = input("Quick log: ")
                response = system.process_message_with_context(message)
                if response.get("logged"):
                    print(f"✅ Logged to: {', '.join(response['logged'])}")
                if response.get("context_analysis"):
                    ctx = response["context_analysis"]
                    print(f"   [Matches: {ctx['specific_matches']} specific, {ctx['general_matches']} general]")

            elif command == "query":
                query = input("Query: ")
                answer = system.query_with_logs(query)
                print(f"\nAnswer: {answer}")

            elif command == "propose":
                conversation = input("Describe activity to track: ")
                proposed = system.propose_new_log_category(conversation)
                if proposed:
                    print(f"✅ Created new category: {proposed['name']}")
                    print(f"   Description: {proposed['description']}")
                    print(f"   Fields: {list(proposed['fields'].keys())}")
                else:
                    print("No new category needed")

            elif command == "patterns":
                # Analyze patterns using adaptive scheduler
                print("Analyzing behavioral patterns...")
                patterns = system.analyze_behavioral_patterns(days=30)
                print(f"\n📊 Pattern Analysis:")
                print(f"   Activity patterns: {len(patterns.get('activity_patterns', []))}")
                print(f"   Productivity cycles: {patterns.get('productivity_cycles', 'Unknown')}")
                print(f"   Most active time: {patterns.get('most_active_time', 'Unknown')}")

            elif command == "schedule":
                # Show today's schedule using adaptive scheduler
                schedule = system.generate_daily_schedule()
                print(f"\n📅 Today's Schedule:")
                for item in schedule:
                    print(f"   {item.scheduled_time.strftime('%H:%M')} - {item.check_in_type.value}")

            elif command == "summary":
                summary = system.get_daily_summary_with_logs()
                print(f"\n📊 Daily Summary for {summary['date']}")
                print(f"   Total logs: {summary['total_logs_today']}")
                print(f"   Categories active: {summary['log_categories_active']}")

                for category, info in summary['log_books'].items():
                    print(f"   {category}: {info['count']} entries")

                print(f"\n   Productivity Score: {summary['activity_metrics']['productivity_score']:.2f}")
                print(f"   Wellness Score: {summary['activity_metrics']['wellness_score']:.2f}")
                print(f"   Character Level: {summary['rpg']['level']}")

                # Show two-tier harmonization status
                harm = summary['harmonization']
                two_tier = harm.get('two_tier_summary', {})
                print(f"\n   Label Groups: {two_tier.get('general_groups', 0)} general, {two_tier.get('specific_groups', 0)} specific")
                if harm['needs_update']:
                    print(f"   ⚠️ Harmonization needed")

            elif command == "harmonize":
                insights = system.get_harmonization_insights()
                print("\n📏 Harmonization Status:")
                for category, stats in insights['categories'].items():
                    print(f"\n{category.upper()}:")
                    print(f"  Total groups: {stats['canonical_count']}")
                    print(f"  Unique concepts: {stats['unique_concepts']}")

                    tier_info = stats.get('two_tier', {})
                    print(f"  Two-tier organization:")
                    print(f"    General groups (>0.5): {tier_info.get('general_groups', 0)}")
                    print(f"    Specific groups (>0.8): {tier_info.get('specific_groups', 0)}")

                    if tier_info.get('general_groups', 0) > 0 and tier_info.get('specific_groups', 0) > 0:
                        ratio = tier_info['specific_groups'] / tier_info['general_groups']
                        print(f"    Specificity ratio: {ratio:.2f}")

                if insights['needs_harmonization']:
                    print(f"\n⚠️ {system.labels_created_count - system.last_harmonization_count} labels need harmonization")
                    print("   Use 'rebuild' command to update two-tier groups")

            elif command == "rebuild":
                confirm = input("Rebuild two-tier harmonization groups? (y/n): ")
                if confirm.lower() == 'y':
                    stats = system.rebuild_two_tier_groups()
                    print("\n📊 Rebuild complete!")
                    print("   Files updated:")
                    print("     - harmonization_general.json")
                    print("     - harmonization_specific.json")
                    print("     - harmonization_report.json")
            
            else:
                print("Unknown command")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    print("\n👋 Keep tracking, keep growing!")


if __name__ == "__main__":
    main()