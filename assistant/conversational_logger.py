"""
Conversational AI Logger for natural tracking interactions.

This module provides a conversational interface to the unified logger,
using LLMs to naturally collect tracking data through friendly conversation.
It coordinates between different models for different purposes:
- Data extraction (gpt-5-mini/nano)
- Conversation management (gpt-4o)
"""

from __future__ import annotations
import json
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

from .logger.unified import UnifiedLogger


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


class CheckInType(Enum):
    """Types of check-ins with different conversation durations."""
    MORNING = "morning"  # 15-30 min comprehensive
    PERIODIC = "periodic"  # 5 min quick status
    EVENING = "evening"  # 10-15 min reflection
    FOCUS = "focus"  # 2 min ultra-quick


class ConversationalLogger:
    """
    AI-powered conversational interface for the unified logger.
    Manages natural dialogue while extracting structured tracking data.
    """

    # System prompts for different models/roles
    EXTRACTOR_PROMPT = """You are a health and productivity data extraction assistant. 
    Your task is to analyze conversational responses and extract structured tracking data.
    
    Extract any mentions of:
    - Nutrition (food, drinks, supplements)
    - Exercise (activities, duration, intensity)
    - Mood (emotional state, stress levels)
    - Sleep (quality, duration, interruptions)
    - Tasks (completed, planned, blocked)
    - Energy levels (1-10 scale or descriptive)
    - Focus levels (1-10 scale or descriptive)
    - Social interactions (meetings, conversations)
    
    Also identify potential new tracking categories the user might benefit from.
    
    Output JSON only with this structure:
    {
        "entries": [
            {
                "category": "nutrition|exercise|mood|sleep|task|energy|focus|social|custom",
                "value": <appropriate value>,
                "confidence": 0.0-1.0,
                "metadata": {}
            }
        ],
        "suggested_categories": ["category_name", ...],
        "extracted_context": "relevant context for conversation"
    }
    """

    CONVERSATIONALIST_PROMPT_TEMPLATE = """You are a supportive personal assistant helping someone track their health and productivity.
    
    Current check-in type: {check_in_type}
    Time budget: {time_budget} minutes
    Time elapsed: {elapsed} minutes
    
    Your goals:
    1. Have a natural, friendly conversation
    2. Gently probe for tracking data in these areas: {categories}
    3. Be encouraging and supportive
    4. Respect the time budget - start wrapping up at 80% of time
    5. Don't be pushy or invasive - if they seem busy, keep it brief
    
    Context from previous interactions:
    {context}
    
    Current data gaps to explore (if natural):
    {data_gaps}
    
    Remember: Quality of interaction > Quantity of data collected
    """

    def __init__(
        self,
        unified_logger: UnifiedLogger,
        openai_client: Any,
        extractor_model: str = "gpt-5-mini",
        conversationalist_model: str = "gpt-4o"
    ):
        """
        Initialize the conversational logger.

        Args:
            unified_logger: The UnifiedLogger instance for storing data
            openai_client: OpenAI client for LLM interactions
            extractor_model: Model for data extraction (efficiency focused)
            conversationalist_model: Model for conversation (rapport focused)
        """
        self.logger = unified_logger
        self.openai_client = openai_client
        self.extractor_model = extractor_model
        self.conversationalist_model = conversationalist_model

        # Track conversation state
        self.current_session = None
        self.conversation_history = []

    def start_check_in(
        self,
        check_in_type: Union[CheckInType, str] = CheckInType.PERIODIC,
        custom_prompts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Start a new check-in conversation.

        Args:
            check_in_type: Type of check-in determining duration and focus
            custom_prompts: Optional specific topics to explore

        Returns:
            Session info including greeting and time budget
        """
        if isinstance(check_in_type, str):
            check_in_type = CheckInType(check_in_type)

        # Set time budgets
        time_budgets = {
            CheckInType.MORNING: (15, 30),  # min, max minutes
            CheckInType.PERIODIC: (5, 10),
            CheckInType.EVENING: (10, 15),
            CheckInType.FOCUS: (2, 5)
        }

        min_time, max_time = time_budgets[check_in_type]

        # Determine focus categories based on check-in type
        if check_in_type == CheckInType.MORNING:
            categories = ["sleep", "mood", "energy", "nutrition", "tasks"]
        elif check_in_type == CheckInType.PERIODIC:
            categories = ["energy", "focus", "tasks", "nutrition"]
        elif check_in_type == CheckInType.EVENING:
            categories = ["tasks", "mood", "social", "exercise", "energy"]
        else:  # FOCUS
            categories = ["focus", "energy"]

        # Get recent context
        context = self._get_recent_context()
        data_gaps = self._identify_data_gaps()

        # Create session with timezone-aware timestamp
        self.current_session = {
            "id": str(ensure_timezone_aware(datetime.now(timezone.utc)).timestamp()),
            "type": check_in_type,
            "started_at": ensure_timezone_aware(datetime.now(timezone.utc)),
            "min_time": min_time,
            "max_time": max_time,
            "categories": categories,
            "context": context,
            "data_gaps": data_gaps,
            "messages": [],
            "extracted_entries": []
        }

        # Generate initial greeting
        greeting = self._generate_greeting(check_in_type, context)

        self.current_session["messages"].append({
            "role": "assistant",
            "content": greeting
        })

        return {
            "session_id": self.current_session["id"],
            "greeting": greeting,
            "time_budget": f"{min_time}-{max_time} minutes",
            "focus_areas": categories
        }

    def process_response(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user response in the conversation.

        Args:
            user_message: The user's message

        Returns:
            Response with next message and any extracted data
        """
        if not self.current_session:
            return {"error": "No active session. Call start_check_in first."}

        # Add user message to history
        self.current_session["messages"].append({
            "role": "user",
            "content": user_message
        })

        # Extract data from the message
        extracted = self._extract_data(user_message)

        # Log the extracted entries immediately
        for entry in extracted.get("entries", []):
            try:
                # Add timestamp if not present (timezone-aware)
                if "timestamp" not in entry:
                    entry["timestamp"] = ensure_timezone_aware(datetime.now(timezone.utc)).isoformat()

                # Log to unified logger
                node = self.logger.log(entry)
                self.current_session["extracted_entries"].append(node)
            except Exception as e:
                print(f"Error logging entry: {e}")

        # Store suggested categories for later review
        if extracted.get("suggested_categories"):
            self._store_category_suggestions(extracted["suggested_categories"])

        # Check if we should wrap up (with timezone-aware comparison)
        now = ensure_timezone_aware(datetime.now(timezone.utc))
        started = ensure_timezone_aware(self.current_session["started_at"])
        elapsed = (now - started).seconds / 60
        should_wrap = elapsed >= self.current_session["min_time"] * 0.8

        # Generate conversational response
        ai_response = self._generate_response(
            self.current_session["messages"],
            extracted.get("extracted_context", ""),
            should_wrap,
            elapsed
        )

        # Add AI response to history
        self.current_session["messages"].append({
            "role": "assistant",
            "content": ai_response
        })

        # Check if conversation should end
        is_complete = (
            should_wrap and self._is_natural_ending(ai_response)
        ) or elapsed >= self.current_session["max_time"]

        response = {
            "message": ai_response,
            "extracted_count": len(extracted.get("entries", [])),
            "session_time": f"{elapsed:.1f} minutes",
            "is_complete": is_complete
        }

        if is_complete:
            response["summary"] = self._complete_session()

        return response

    def _extract_data(self, message: str) -> Dict[str, Any]:
        """Extract structured data from a conversational message."""
        try:
            # The app client needs messages format for chat models
            messages = [
                {"role": "system", "content": self.EXTRACTOR_PROMPT},
                {"role": "user", "content": f"User message: {message}"}
            ]

            # The OpenAI client returns a string directly
            response = self.openai_client.chat(
                messages=messages,
                model=self.extractor_model,
                response_format={"type": "json_object"}  # Ensure JSON response
            )

            # Handle the response
            if isinstance(response, str):
                content = response.strip()
            else:
                # Fallback if response structure changes
                content = str(response).strip()

            # FIX: Check if content is empty before trying to parse
            if not content:
                print("Warning: Empty response from OpenAI client")
                return {"entries": [], "suggested_categories": []}

            # Clean up the response
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'\s*```$', '', content)

            # Check again after cleaning
            if not content:
                print("Warning: Empty content after cleaning")
                return {"entries": [], "suggested_categories": []}

            # Parse the JSON response
            result = json.loads(content)

            # Ensure required fields exist
            if "entries" not in result:
                result["entries"] = []
            if "suggested_categories" not in result:
                result["suggested_categories"] = []

            return result

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response was: {content if 'content' in locals() else 'N/A'}")
            # Try to extract something useful from the response
            if 'content' in locals() and isinstance(content, str) and content:
                # Try to extract any text as a note
                return {
                    "entries": [{"category": "note", "value": content.strip()}],
                    "suggested_categories": []
                }
            return {"entries": [], "suggested_categories": []}
        except Exception as e:
            print(f"Extraction error: {e}")
            return {"entries": [], "suggested_categories": []}
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response was: {content if 'content' in locals() else 'N/A'}")
            # Try to extract something useful from the response
            if 'content' in locals() and isinstance(content, str):
                # Try to extract any text as a note
                return {
                    "entries": [{"category": "note", "value": content.strip()}],
                    "suggested_categories": []
                }
            return {"entries": [], "suggested_categories": []}
        except Exception as e:
            print(f"Extraction error: {e}")
            return {"entries": [], "suggested_categories": []}

    def _generate_response(
            self,
            messages: List[Dict],
            context: str,
            should_wrap: bool,
            elapsed: float
    ) -> str:
        """Generate conversational response."""
        try:
            # Build prompt
            check_in_type = self.current_session["type"].value
            time_budget = self.current_session["max_time"]
            categories = ", ".join(self.current_session["categories"])
            data_gaps = ", ".join(self.current_session.get("data_gaps", []))

            system_prompt = self.CONVERSATIONALIST_PROMPT_TEMPLATE.format(
                check_in_type=check_in_type,
                time_budget=time_budget,
                elapsed=elapsed,
                categories=categories,
                context=self.current_session.get("context", "None available"),
                data_gaps=data_gaps or "None identified"
            )

            if should_wrap:
                system_prompt += "\n\nStart wrapping up the conversation naturally."

            # Get response from conversationalist model
            all_messages = [
                {"role": "system", "content": system_prompt},
                *messages
            ]

            # The OpenAI client returns a string directly
            response = self.openai_client.chat(
                messages=all_messages,
                model=self.conversationalist_model
            )

            # FIX: Check for empty or None response
            if response is None:
                print("Warning: None response from OpenAI client")
                if should_wrap:
                    return "Thanks for checking in! Let me know if you need anything else."
                else:
                    return "I understand. How else are you feeling right now?"

            # Handle the response
            if isinstance(response, str):
                result = response.strip()
            else:
                result = str(response).strip()

            # FIX: Check if result is empty
            if not result:
                print("Warning: Empty response from conversationalist model")
                if should_wrap:
                    return "Thanks for checking in! Let me know if you need anything else."
                else:
                    return "I understand. How else are you feeling right now?"

            return result

        except Exception as e:
            print(f"Error generating response: {e}")
            # Return a fallback response
            if should_wrap:
                return "Thanks for checking in! Let me know if you need anything else."
            else:
                return "I understand. How else are you feeling right now?"

    def _generate_greeting(self, check_in_type: CheckInType, context: str) -> str:
        """Generate an appropriate greeting for the check-in type."""
        time_of_day = ensure_timezone_aware(datetime.now(timezone.utc)).strftime("%H:%M")

        greetings = {
            CheckInType.MORNING: [
                f"Good morning! How did you sleep?",
                f"Morning! Ready to plan your day?",
                f"Hey there! How are you feeling this morning?"
            ],
            CheckInType.PERIODIC: [
                f"Quick check - how's your energy right now?",
                f"Hey! What are you working on?",
                f"Hi! How's the focus level?"
            ],
            CheckInType.EVENING: [
                f"Evening! How was your day?",
                f"Hey! Ready for a quick daily reflection?",
                f"Hi there! What were today's wins?"
            ],
            CheckInType.FOCUS: [
                f"Quick pulse check - focus level 1-10?",
                f"Hey - need anything to stay in flow?",
                f"Two seconds - energy check?"
            ]
        }

        import random
        base_greeting = random.choice(greetings.get(check_in_type, ["Hello!"]))

        # Add context if relevant
        if context and "streak" in context:
            base_greeting += f" (Great job on your {context['streak']} day streak!)"

        return base_greeting

    def _get_recent_context(self) -> str:
        """Get context from recent logs."""
        try:
            # Get today's summary with timezone-aware date
            today = ensure_timezone_aware(datetime.now(timezone.utc)).date()
            summary = self.logger.get_daily_summary(str(today))

            context_items = []
            if "sleep" in summary:
                context_items.append(f"Sleep entries: {summary['sleep']['count']}")
            if "exercise" in summary:
                context_items.append(f"Exercise logged today")
            if "mood" in summary and summary["mood"].get("avg"):
                context_items.append(f"Avg mood: {summary['mood']['avg']:.1f}")

            return "; ".join(context_items) if context_items else "First check-in today"
        except Exception:
            return "Starting fresh"

    def _identify_data_gaps(self) -> List[str]:
        """Identify what data hasn't been collected recently."""
        try:
            today = ensure_timezone_aware(datetime.now(timezone.utc)).date()
            summary = self.logger.get_daily_summary(str(today))

            expected = {"nutrition", "exercise", "mood", "sleep", "energy", "focus"}
            logged = set(summary.keys())

            return list(expected - logged)
        except Exception:
            return []

    def _is_natural_ending(self, message: str) -> bool:
        """Check if the message seems like a natural conversation ending."""
        ending_phrases = [
            "take care", "talk later", "have a great",
            "have a good", "catch you", "see you",
            "goodbye", "bye", "thanks for checking in",
            "all set", "that's all"
        ]

        message_lower = message.lower()
        return any(phrase in message_lower for phrase in ending_phrases)

    def _complete_session(self) -> Dict[str, Any]:
        """Complete the current session and return summary."""
        if not self.current_session:
            return {}

        # Use timezone-aware datetime for duration calculation
        now = ensure_timezone_aware(datetime.now(timezone.utc))
        started = ensure_timezone_aware(self.current_session["started_at"])

        summary = {
            "session_id": self.current_session["id"],
            "duration_minutes": (now - started).seconds / 60,
            "entries_logged": len(self.current_session["extracted_entries"]),
            "categories_covered": list(set(
                e["attrs"]["category"]
                for e in self.current_session["extracted_entries"]
                if "attrs" in e and "category" in e["attrs"]
            )),
            "message_count": len([
                m for m in self.current_session["messages"]
                if m["role"] == "user"
            ])
        }

        # Archive session
        self.conversation_history.append(self.current_session)
        self.current_session = None

        return summary

    def _store_category_suggestions(self, suggestions: List[str]) -> None:
        """Store category suggestions for later review."""
        # In production, this would persist to a database
        # For now, just log them
        for suggestion in suggestions:
            print(f"Suggested new category: {suggestion}")

    def quick_log(self, message: str) -> Dict[str, Any]:
        """
        Quick one-shot logging without full conversation.
        Useful for voice notes or quick inputs.

        Args:
            message: Natural language description of what to log

        Returns:
            Summary of what was logged
        """
        try:
            extracted = self._extract_data(message)
        except Exception as e:
            print(f"Extraction failed: {e}")
            extracted = {"entries": [], "suggested_categories": []}
        
        logged = []
        entries = extracted.get("entries", [])
        
        # Ensure entries is a list
        if not isinstance(entries, list):
            if isinstance(entries, str):
                # If it's a string, wrap it in a list as a simple entry
                entries = [{"category": "note", "value": entries}]
            else:
                entries = []
        
        for entry in entries:
            try:
                # Handle different entry formats
                if isinstance(entry, str):
                    # Convert string to dict
                    entry = {
                        "category": "note",
                        "value": entry,
                        "confidence": 0.5
                    }
                elif isinstance(entry, dict):
                    # Ensure required fields exist
                    if "category" not in entry:
                        entry["category"] = "unknown"
                    if "value" not in entry and len(entry) > 0:
                        # Use the first non-category field as value
                        for key, val in entry.items():
                            if key != "category":
                                entry["value"] = val
                                break
                        else:
                            entry["value"] = str(entry)
                else:
                    # Skip non-string, non-dict entries
                    print(f"Skipping invalid entry type: {type(entry)}")
                    continue
                
                # Add timestamp if not present
                if "timestamp" not in entry:
                    entry["timestamp"] = ensure_timezone_aware(datetime.now(timezone.utc)).isoformat()
                
                # Try to log the entry
                node = self.logger.log(entry)
                logged.append({
                    "category": entry.get("category", "unknown"),
                    "value": entry.get("value", "")
                })
                
            except Exception as e:
                print(f"Quick log entry error: {e}")
                print(f"Problematic entry: {entry}")
                continue

        return {
            "logged_count": len(logged),
            "entries": logged,
            "suggestions": extracted.get("suggested_categories", [])
        }
