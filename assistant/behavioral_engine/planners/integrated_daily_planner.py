"""
Integrated Daily Planner and Logger System
==========================================

Combines context-aware scheduling, feature extraction, and daily planning
into a cohesive system for behavioral tracking and assistance.
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timedelta, timezone, time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

# Import app components
from app.openai_client import OpenAIClient
from app.label_integration_wrappers import LabelGenerator

# Import assistant components
from assistant.logger.unified import UnifiedLogger
from assistant.graph.store import GraphStore
from assistant.conversational_logger import CheckInType
from assistant.logger.temporal_personal_profile import TemporalPersonalProfile

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
class DailyPlan:
    """Represents a daily plan with scheduled activities."""
    
    date: datetime
    check_ins: List[Dict] = field(default_factory=list)
    reminders: List[Dict] = field(default_factory=list)
    tasks: List[Dict] = field(default_factory=list)
    menu: Optional[Dict] = None
    notes: str = ""
    
    def add_check_in(self, time: datetime, check_type: CheckInType, context: Dict = None):
        """Add a check-in to the plan."""
        self.check_ins.append({
            "time": time.isoformat(),
            "type": check_type.value,
            "context": context or {},
            "completed": False
        })
    
    def add_reminder(self, time: datetime, title: str, description: str = ""):
        """Add a reminder to the plan."""
        self.reminders.append({
            "time": time.isoformat(),
            "title": title,
            "description": description,
            "acknowledged": False
        })
    
    def add_task(self, title: str, priority: int = 1, estimated_minutes: int = 30):
        """Add a task to the plan."""
        self.tasks.append({
            "title": title,
            "priority": priority,
            "estimated_minutes": estimated_minutes,
            "completed": False,
            "actual_minutes": None
        })
    
    def set_menu(self, breakfast: str = "", lunch: str = "", dinner: str = "", snacks: List[str] = None):
        """Set the meal plan for the day."""
        self.menu = {
            "breakfast": breakfast,
            "lunch": lunch,
            "dinner": dinner,
            "snacks": snacks or []
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "date": self.date.isoformat(),
            "check_ins": self.check_ins,
            "reminders": self.reminders,
            "tasks": self.tasks,
            "menu": self.menu,
            "notes": self.notes
        }


class IntegratedDailyPlanner:
    """
    Main system that integrates all components for daily planning
    and behavioral tracking.
    """
    
    def __init__(
        self,
        data_dir: str = "data/planner",
        labels_dir: str = "labels",
        chunks_dir: str = "data/chunks",
        use_sandbox: bool = False,
        sandbox_name: Optional[str] = None
    ):
        """Initialize the integrated planner."""
        self.data_dir = data_dir
        self.labels_dir = labels_dir
        self.chunks_dir = chunks_dir
        
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize core components
        self.openai_client = OpenAIClient()
        self.logger = UnifiedLogger(data_dir="data/tracking")
        self.graph = GraphStore(data_dir="data/graph")
        self.profile = TemporalPersonalProfile()
        
        # Import and initialize scheduler from previous artifact
        # Update these three imports
        from assistant.behavioral_engine.schedulers.context_aware_scheduler import ContextAwareScheduler
        self.scheduler = ContextAwareScheduler(
            logger=self.logger,
            graph_store=self.graph,
            openai_client=self.openai_client,
            labels_dir=labels_dir,
            chunks_dir=chunks_dir,
            sandbox_mode=use_sandbox,
            sandbox_name=sandbox_name
        )
        
        # Import and initialize feature extractor from previous artifact
        from assistant.behavioral_engine.features.enhanced_feature_extraction import EnhancedFeatureExtractor
        self.feature_extractor = EnhancedFeatureExtractor(
            logger=self.logger,
            graph_store=self.graph
        )
        
        # Import RPG system from previous artifact
        from assistant.behavioral_engine.gamification.rpg_system import RPGSystem
        self.rpg = RPGSystem(
            logger=self.logger,
            graph_store=self.graph
        )
        
        # Current state
        self.current_plan = None
        self.active_session = None
        self.daily_stats = {}
    
    def create_daily_plan(
        self,
        date: Optional[datetime] = None,
        custom_schedule: Optional[Dict] = None
    ) -> DailyPlan:
        """
        Create a daily plan with check-ins and reminders.
        
        Args:
            date: Date for the plan (default: today)
            custom_schedule: Optional custom schedule overrides
            
        Returns:
            Daily plan
        """
        if date is None:
            date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        
        plan = DailyPlan(date=date)
        
        # Default schedule (can be customized)
        default_schedule = {
            "morning_check_in": time(7, 30),
            "morning_l_theanine": time(8, 0),
            "mid_morning_focus": time(10, 30),
            "lunch_reminder": time(12, 0),
            "afternoon_check_in": time(14, 30),
            "afternoon_l_theanine": time(14, 0),
            "exercise_reminder": time(17, 0),
            "evening_check_in": time(20, 30),
            "wind_down": time(21, 30)
        }
        
        # Merge with custom schedule
        if custom_schedule:
            default_schedule.update(custom_schedule)
        
        # Add check-ins
        plan.add_check_in(
            ensure_timezone_aware(datetime.combine(date.date()), default_schedule["morning_check_in"]),
            CheckInType.MORNING,
            {"topics": ["sleep", "mood", "energy", "planning"]}
        )
        
        plan.add_check_in(
            ensure_timezone_aware(datetime.combine(date.date()), default_schedule["mid_morning_focus"]),
            CheckInType.FOCUS,
            {"topics": ["focus", "productivity"]}
        )
        
        plan.add_check_in(
            ensure_timezone_aware(datetime.combine(date.date()), default_schedule["afternoon_check_in"]),
            CheckInType.PERIODIC,
            {"topics": ["energy", "progress", "blockers"]}
        )
        
        plan.add_check_in(
            ensure_timezone_aware(datetime.combine(date.date()), default_schedule["evening_check_in"]),
            CheckInType.EVENING,
            {"topics": ["accomplishments", "social", "reflection"]}
        )
        
        # Add medication reminders
        plan.add_reminder(
            ensure_timezone_aware(datetime.combine(date.date()), default_schedule["morning_l_theanine"]),
            "L-theanine with morning coffee",
            "Take 200mg L-theanine with your coffee for calm focus"
        )
        
        plan.add_reminder(
            ensure_timezone_aware(datetime.combine(date.date()), default_schedule["afternoon_l_theanine"]),
            "L-theanine with afternoon coffee",
            "Optional: Take L-theanine if having afternoon coffee"
        )
        
        # Add activity reminders
        plan.add_reminder(
            ensure_timezone_aware(datetime.combine(date.date()), default_schedule["lunch_reminder"]),
            "Lunch break",
            "Time to eat and recharge"
        )
        
        plan.add_reminder(
            ensure_timezone_aware(datetime.combine(date.date()), default_schedule["exercise_reminder"]),
            "Exercise time",
            "Movement break or workout session"
        )
        
        plan.add_reminder(
            ensure_timezone_aware(datetime.combine(date.date()), default_schedule["wind_down"]),
            "Wind down routine",
            "Start winding down for better sleep"
        )
        
        # Generate daily quests
        quests = self.rpg.generate_daily_quests()
        for quest in quests:
            plan.add_task(
                title=quest.name,
                priority=2,
                estimated_minutes=30
            )
        
        # Save plan
        self.current_plan = plan
        self._save_plan(plan)
        
        return plan
    
    def start_check_in(self, check_type: Optional[CheckInType] = None) -> Dict:
        """
        Start a check-in session.
        
        Args:
            check_type: Type of check-in (auto-detects if not specified)
            
        Returns:
            Session information
        """
        # Auto-detect check-in type based on time if not specified
        if check_type is None:
            check_type = self._detect_check_in_type()
        
        # Build context from profile and recent activity
        context = self._build_check_in_context(check_type)
        
        # Start session with context-aware scheduler
        session = self.scheduler.start_check_in(check_type, context)
        self.active_session = session
        
        # Log check-in start
        self.logger.log({
            "category": "check_in",
            "value": f"Started {check_type.value} check-in",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {"session_id": session["session_id"]}
        })
        
        return session
    
    def process_message(self, message: str) -> Dict:
        """
        Process a message in the active check-in.
        
        Args:
            message: User message
            
        Returns:
            Response with extracted data and features
        """
        if not self.active_session:
            return {"error": "No active session. Start a check-in first."}
        
        # Process with context-aware scheduler
        response = self.scheduler.process_message(message)
        
        # Extract features from message
        timestamp = datetime.now(timezone.utc)
        message_features = self.feature_extractor.extract_message_features(
            message,
            timestamp
        )
        
        # Process activity for RPG system
        if response.get("extracted_count", 0) > 0:
            # Award XP for logging
            rpg_result = self.rpg.process_activity(
                "check_in",
                1,
                {"messages": 1, "extracted": response["extracted_count"]}
            )
            
            response["xp_gained"] = rpg_result.get("xp_gained", 0)
            response["achievements"] = rpg_result.get("achievements_unlocked", [])
        
        # Update daily stats
        self._update_daily_stats(message_features, response)
        
        # Add features to response
        response["features"] = {
            "vocabulary_richness": message_features.vocabulary_richness,
            "rare_word_ratio": message_features.rare_word_ratio,
            "engagement": message_features.word_count > 20  # Simple engagement metric
        }
        
        return response
    
    def complete_check_in(self) -> Dict:
        """
        Complete the active check-in session.
        
        Returns:
            Session summary with features
        """
        if not self.active_session:
            return {"error": "No active session to complete"}
        
        session_id = self.active_session["session_id"]
        
        # Get session messages from scheduler
        messages = self.scheduler.current_context.temporal_messages
        labels = []  # Would be extracted from label files
        
        # Extract session features
        if messages:
            session_features = self.feature_extractor.extract_session_features(
                session_id,
                messages,
                labels
            )
            
            # Save features
            self.feature_extractor.save_features(
                session_features,
                f"session_{session_id}_features.json"
            )
            
            # Get feature summary
            feature_summary = self.feature_extractor.get_feature_summary(session_features)
        else:
            feature_summary = {}
        
        # Mark check-in as complete in plan
        if self.current_plan:
            for check_in in self.current_plan.check_ins:
                if not check_in["completed"] and check_in["type"] == self.active_session.get("check_in_type"):
                    check_in["completed"] = True
                    check_in["completed_at"] = datetime.now(timezone.utc).isoformat()
                    break
            
            self._save_plan(self.current_plan)
        
        # Log completion
        self.logger.log({
            "category": "check_in",
            "value": f"Completed check-in",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "session_id": session_id,
                "duration_minutes": feature_summary.get("session", {}).get("duration"),
                "messages": len(messages)
            }
        })
        
        # Clear active session
        self.active_session = None
        
        return {
            "session_id": session_id,
            "summary": feature_summary,
            "daily_progress": self.get_daily_progress()
        }
    
    def log_quick_entry(self, entry_text: str) -> Dict:
        """
        Quick logging without a full check-in session.
        
        Args:
            entry_text: Natural language entry
            
        Returns:
            Extraction results
        """
        # Extract structured data
        extracted = self.scheduler._extract_structured_data(entry_text)
        
        # Log entries
        logged = []
        for entry in extracted.get("entries", []):
            if "timestamp" not in entry:
                entry["timestamp"] = datetime.now(timezone.utc).isoformat()
            
            node = self.logger.log(entry)
            logged.append({
                "category": entry["category"],
                "value": entry.get("value")
            })
            
            # Process for RPG
            self.rpg.process_activity(
                entry["category"],
                entry.get("value", 1),
                entry.get("metadata", {})
            )
        
        # Generate and store labels
        chunk_id = self.scheduler._store_message_chunk({
            "role": "user",
            "content": entry_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "quick_log": True
        })
        
        labels = self.scheduler._generate_and_store_labels(entry_text, chunk_id)
        
        return {
            "logged": logged,
            "labels": [l["label"] for l in labels],
            "xp_gained": len(logged) * 5  # Simple XP calculation
        }
    
    def generate_menu_suggestions(self) -> Dict:
        """
        Generate menu suggestions based on pantry and preferences.
        
        Returns:
            Menu suggestions for the day
        """
        # Get dietary preferences from profile
        dietary_facts = self.profile.get_facts_by_category("health")
        preferences = []
        restrictions = []
        
        for fact in dietary_facts:
            if "diet" in fact.get("fact", "").lower():
                preferences.append(fact["fact"])
            if "allerg" in fact.get("fact", "").lower() or "intoleran" in fact.get("fact", "").lower():
                restrictions.append(fact["fact"])
        
        # Get recent meals to avoid repetition
        recent_meals = self._get_recent_meals(days=3)
        
        # Generate suggestions using OpenAI
        prompt = f"""Generate healthy meal suggestions for today.
        
        Dietary preferences: {', '.join(preferences) if preferences else 'None specified'}
        Restrictions: {', '.join(restrictions) if restrictions else 'None'}
        Recent meals to avoid: {', '.join(recent_meals) if recent_meals else 'None'}
        
        Suggest:
        - Breakfast (quick, energizing)
        - Lunch (balanced, satisfying)
        - Dinner (nutritious, not too heavy)
        - 2 healthy snacks
        
        Keep suggestions simple and practical.
        Format as JSON with keys: breakfast, lunch, dinner, snacks (array)
        """
        
        try:
            response = self.openai_client.chat(
                messages=[
                    {"role": "system", "content": "You are a helpful nutrition assistant."},
                    {"role": "user", "content": prompt}
                ],
                model="gpt-4o"
            )
            
            # Parse response
            content = response.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            
            menu = json.loads(content)
            
            # Save to plan
            if self.current_plan:
                self.current_plan.set_menu(**menu)
                self._save_plan(self.current_plan)
            
            return menu
            
        except Exception as e:
            print(f"Error generating menu: {e}")
            return {
                "breakfast": "Oatmeal with berries",
                "lunch": "Salad with protein",
                "dinner": "Grilled vegetables with quinoa",
                "snacks": ["Apple with almond butter", "Greek yogurt"]
            }
    
    def get_daily_progress(self) -> Dict:
        """Get progress for the current day."""
        if not self.current_plan:
            return {"error": "No plan for today"}
        
        # Count completed items
        check_ins_total = len(self.current_plan.check_ins)
        check_ins_completed = sum(1 for c in self.current_plan.check_ins if c["completed"])
        
        tasks_total = len(self.current_plan.tasks)
        tasks_completed = sum(1 for t in self.current_plan.tasks if t["completed"])
        
        reminders_total = len(self.current_plan.reminders)
        reminders_acknowledged = sum(1 for r in self.current_plan.reminders if r["acknowledged"])
        
        # Get RPG summary
        rpg_summary = self.rpg.get_character_summary()
        
        # Get activity summary for today
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        today_activity = self.feature_extractor.extract_activity_features(
            today_start,
            datetime.now(timezone.utc)
        )
        
        return {
            "date": self.current_plan.date.strftime("%Y-%m-%d"),
            "check_ins": {
                "completed": check_ins_completed,
                "total": check_ins_total,
                "percentage": (check_ins_completed / check_ins_total * 100) if check_ins_total > 0 else 0
            },
            "tasks": {
                "completed": tasks_completed,
                "total": tasks_total,
                "percentage": (tasks_completed / tasks_total * 100) if tasks_total > 0 else 0
            },
            "reminders": {
                "acknowledged": reminders_acknowledged,
                "total": reminders_total
            },
            "rpg": {
                "level": rpg_summary["character"]["level"],
                "xp_today": self.daily_stats.get("xp_gained", 0),
                "achievements_today": self.daily_stats.get("achievements_unlocked", [])
            },
            "activity": {
                "logs": today_activity.total_logs,
                "categories": today_activity.unique_categories,
                "focus_score": today_activity.avg_focus_score,
                "productivity_score": self.feature_extractor._calculate_productivity_score(today_activity)
            },
            "menu_planned": self.current_plan.menu is not None
        }
    
    def _detect_check_in_type(self) -> CheckInType:
        """Auto-detect appropriate check-in type based on time."""
        current_hour = datetime.now(timezone.utc).hour
        
        if 5 <= current_hour < 10:
            return CheckInType.MORNING
        elif 10 <= current_hour < 12 or 14 <= current_hour < 16:
            return CheckInType.PERIODIC
        elif 18 <= current_hour < 22:
            return CheckInType.EVENING
        else:
            return CheckInType.FOCUS

    def _build_check_in_context(self, check_type: CheckInType) -> Dict[str, Any]:
        """Build context for check-in based on profile and recent activity."""
        # Explicitly type hint as Dict[str, Any] to allow mixed types
        context: Dict[str, Any] = {
            "check_type": check_type.value,
            "time": datetime.now(timezone.utc).isoformat()
        }

        # Now PyCharm knows context can hold any type of value
        if check_type == CheckInType.MORNING:
            routines = self.profile.get_facts_by_category("routines")
            # This is now valid - storing a list in the dict
            context["routines"] = [f.get("fact", f.get("value", "")) for f in routines[:3]]

        elif check_type == CheckInType.EVENING:
            goals = self.profile.get_facts_by_category("goals")
            # This is now valid - storing a list in the dict
            context["goals"] = [f.get("fact", f.get("value", "")) for f in goals[:3]]

        # Add recent activity summary - dict is also valid now
        last_hour = datetime.now(timezone.utc) - timedelta(hours=1)
        recent_activity = self.feature_extractor.extract_activity_features(
            last_hour,
            datetime.now(timezone.utc)
        )

        context["recent_activity"] = {
            "logs": recent_activity.total_logs,
            "categories": list(recent_activity.category_counts.keys())
        }

        return context
    
    def _update_daily_stats(self, message_features, response):
        """Update daily statistics."""
        # Initialize if needed
        if "messages_sent" not in self.daily_stats:
            self.daily_stats = {
                "messages_sent": 0,
                "words_total": 0,
                "rare_words_total": 0,
                "xp_gained": 0,
                "achievements_unlocked": []
            }
        
        # Update stats
        self.daily_stats["messages_sent"] += 1
        self.daily_stats["words_total"] += message_features.word_count
        self.daily_stats["rare_words_total"] += message_features.rare_word_count
        
        if "xp_gained" in response:
            self.daily_stats["xp_gained"] += response["xp_gained"]
        
        if "achievements" in response:
            self.daily_stats["achievements_unlocked"].extend(response["achievements"])

    def _get_recent_meals(self, days: int = 3) -> List[str]:
        """Get recent meal entries."""
        # GraphStore.search_nodes doesn't support created_after parameter
        # We need to get all nutrition nodes and filter manually

        start_time = datetime.now(timezone.utc) - timedelta(days=days)

        # Use the correct search_nodes parameters
        nodes = self.graph.search_nodes(
            node_type="log",  # Filter by node type
            limit=100  # Get recent nodes
        )

        meals = []
        for node in nodes:
            # Check if it's a nutrition log and recent enough
            if node.get("attrs", {}).get("category") == "nutrition":
                # Parse the created_at timestamp
                try:
                    created_at = datetime.fromisoformat(
                        node.get("created_at", "").replace("Z", "+00:00")
                    )
                    if created_at >= start_time:
                        value = node.get("attrs", {}).get("value", "")
                        if value and isinstance(value, str):
                            meals.append(value)
                except:
                    continue

        return meals[-10:]  # Last 10 meals
    
    def _save_plan(self, plan: DailyPlan):
        """Save plan to disk."""
        filename = f"plan_{plan.date.strftime('%Y%m%d')}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(plan.to_dict(), f, indent=2)
    
    def _load_plan(self, date: datetime) -> Optional[DailyPlan]:
        """Load plan from disk."""
        filename = f"plan_{date.strftime('%Y%m%d')}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
                plan = DailyPlan(date=ensure_timezone_aware(datetime.fromisoformat(data["date"])))
                plan.check_ins = data.get("check_ins", [])
                plan.reminders = data.get("reminders", [])
                plan.tasks = data.get("tasks", [])
                plan.menu = data.get("menu")
                plan.notes = data.get("notes", "")
                return plan
        
        return None


# CLI Interface for testing
def main():
    """Command-line interface for the planner."""
    import sys
    
    planner = IntegratedDailyPlanner()
    
    print("📅 Integrated Daily Planner")
    print("=" * 40)
    
    # Create or load today's plan
    today = datetime.now(timezone.utc)
    existing_plan = planner._load_plan(today)
    
    if existing_plan:
        planner.current_plan = existing_plan
        print("Loaded existing plan for today")
    else:
        plan = planner.create_daily_plan()
        print(f"Created plan for {plan.date.strftime('%A, %B %d, %Y')}")
    
    # Show daily progress
    progress = planner.get_daily_progress()
    print(f"\n📊 Daily Progress:")
    print(f"  Check-ins: {progress['check_ins']['completed']}/{progress['check_ins']['total']} "
          f"({progress['check_ins']['percentage']:.0f}%)")
    print(f"  Tasks: {progress['tasks']['completed']}/{progress['tasks']['total']} "
          f"({progress['tasks']['percentage']:.0f}%)")
    print(f"  Character: Level {progress['rpg']['level']} "
          f"(+{progress['rpg']['xp_today']:.0f} XP today)")
    
    # Command loop
    while True:
        print("\n" + "=" * 40)
        print("Commands: checkin, quick, menu, progress, sandbox, quit")
        command = input("> ").strip().lower()
        
        if command == "quit":
            break
        
        elif command == "checkin":
            # Start check-in
            session = planner.start_check_in()
            print(f"\n{session['greeting']}")
            
            # Check-in conversation loop
            while planner.active_session:
                user_input = input("You: ")
                if user_input.lower() in ["done", "exit", "quit"]:
                    summary = planner.complete_check_in()
                    print(f"\n✅ Check-in complete!")
                    print(f"Session summary: {json.dumps(summary['summary'], indent=2)}")
                    break
                
                response = planner.process_message(user_input)
                print(f"Assistant: {response['response']}")
                if response.get("xp_gained"):
                    print(f"  [+{response['xp_gained']:.0f} XP]")
        
        elif command == "quick":
            entry = input("Quick log: ")
            result = planner.log_quick_entry(entry)
            print(f"✅ Logged: {result['logged']}")
            print(f"   Labels: {', '.join(result['labels'])}")
            print(f"   [+{result['xp_gained']} XP]")
        
        elif command == "menu":
            print("Generating menu suggestions...")
            menu = planner.generate_menu_suggestions()
            print(f"\n🍽️ Today's Menu:")
            print(f"  Breakfast: {menu['breakfast']}")
            print(f"  Lunch: {menu['lunch']}")
            print(f"  Dinner: {menu['dinner']}")
            print(f"  Snacks: {', '.join(menu['snacks'])}")
        
        elif command == "progress":
            progress = planner.get_daily_progress()
            print(f"\n📊 Daily Progress Update:")
            print(json.dumps(progress, indent=2))
        
        elif command == "sandbox":
            name = input("Sandbox name: ")
            planner.scheduler.create_sandbox(
                name=name,
                ruleset={"type": "experimental", "created": datetime.now(timezone.utc).isoformat()}
            )
            planner.scheduler.switch_to_sandbox(name)
            print(f"✅ Switched to sandbox: {name}")
        
        else:
            print("Unknown command")
    
    print("\n👋 Goodbye! Keep up the great work!")


if __name__ == "__main__":
    main()
