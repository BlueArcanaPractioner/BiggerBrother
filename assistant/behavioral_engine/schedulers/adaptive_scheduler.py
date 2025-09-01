"""
Smart Adaptive Check-in Scheduler
==================================

A scheduling system that learns your patterns and optimally times check-ins,
pomodoro sessions, and reminders based on your behavioral data.
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timedelta, timezone, time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics
from collections import defaultdict, Counter

from assistant.logger.unified import UnifiedLogger
from assistant.graph.store import GraphStore
from assistant.conversational_logger import ConversationalLogger, CheckInType
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
class ActivityPattern:
    """Represents a recurring activity pattern."""
    activity_type: str
    typical_start: time  # Typical start time
    typical_duration: int  # Minutes
    days_of_week: List[int]  # 0=Monday, 6=Sunday
    confidence: float
    last_occurrence: Optional[datetime] = None

    def is_active_on_day(self, day: int) -> bool:
        """Check if pattern is active on given day of week."""
        return day in self.days_of_week


@dataclass
class CheckInSchedule:
    """Scheduled check-in with adaptive timing."""
    check_in_type: CheckInType
    scheduled_time: datetime
    priority: float  # 0.0-1.0
    context: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    skipped: bool = False
    actual_time: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "check_in_type": self.check_in_type.value,
            "scheduled_time": ensure_timezone_aware(self.scheduled_time).isoformat(),
            "priority": self.priority,
            "context": self.context,
            "completed": self.completed,
            "skipped": self.skipped,
            "actual_time": ensure_timezone_aware(self.actual_time).isoformat() if self.actual_time else None
        }


@dataclass
class PomodoroSession:
    """Pomodoro work session with context."""
    start_time: datetime
    duration_minutes: int = 25
    break_minutes: int = 5
    task: Optional[str] = None
    completed: bool = False
    focus_score: Optional[float] = None

    @property
    def end_time(self) -> datetime:
        """Calculate session end time."""
        return ensure_timezone_aware(self.start_time) + timedelta(minutes=self.duration_minutes)


class TransitionType(Enum):
    """Types of activity transitions."""
    WAKE_UP = "wake_up"
    START_WORK = "start_work"
    BREAK = "break"
    MEAL = "meal"
    END_WORK = "end_work"
    EXERCISE = "exercise"
    WIND_DOWN = "wind_down"
    SLEEP = "sleep"


class AdaptiveScheduler:
    """
    Learns from behavioral patterns to optimally schedule check-ins,
    reminders, and interventions.
    """

    def __init__(
        self,
        logger: UnifiedLogger,
        graph_store: GraphStore,
        profile: TemporalPersonalProfile,
        data_dir: str = "data/scheduler"
    ):
        """Initialize the adaptive scheduler."""
        self.logger = logger
        self.graph = graph_store
        self.profile = profile
        self.data_dir = data_dir

        # Create data directory if needed
        os.makedirs(data_dir, exist_ok=True)

        # Load or initialize patterns and schedules
        self.activity_patterns = self._load_patterns()
        self.transition_times = self._load_transition_times()
        self.current_schedule = []
        self.pomodoro_sessions = []

        # Learning parameters
        self.learning_rate = 0.1
        self.pattern_confidence_threshold = 0.7

    def _calculate_optimal_times(self, patterns: Dict) -> Dict[str, datetime]:
        """
        Calculate optimal check-in times based on activity patterns.

        Args:
            patterns: Dictionary containing analyzed patterns

        Returns:
            Dictionary of optimal times for different check-in types
        """
        optimal_times = {}

        # Use timezone-aware base time
        base_time = ensure_timezone_aware(datetime.now(timezone.utc))

        # Find most active hours
        daily_patterns = patterns.get("daily_patterns", {})

        # Morning check-in (6am-10am window)
        morning_activity = {}
        for hour in range(6, 11):
            morning_activity[hour] = len(daily_patterns.get(hour, []))

        if morning_activity:
            best_morning = max(morning_activity, key=morning_activity.get)
            optimal_times["morning"] = base_time.replace(
                hour=best_morning, minute=0, second=0, microsecond=0
            )
        else:
            # Default morning time
            optimal_times["morning"] = base_time.replace(
                hour=8, minute=0, second=0, microsecond=0
            )

        # Afternoon check-in (12pm-4pm window)
        afternoon_activity = {}
        for hour in range(12, 17):
            afternoon_activity[hour] = len(daily_patterns.get(hour, []))

        if afternoon_activity:
            best_afternoon = max(afternoon_activity, key=afternoon_activity.get)
            optimal_times["afternoon"] = base_time.replace(
                hour=best_afternoon, minute=0, second=0, microsecond=0
            )
        else:
            # Default afternoon time
            optimal_times["afternoon"] = base_time.replace(
                hour=14, minute=0, second=0, microsecond=0
            )

        # Evening check-in (6pm-10pm window)
        evening_activity = {}
        for hour in range(18, 23):
            evening_activity[hour] = len(daily_patterns.get(hour, []))

        if evening_activity:
            best_evening = max(evening_activity, key=evening_activity.get)
            optimal_times["evening"] = base_time.replace(
                hour=best_evening, minute=0, second=0, microsecond=0
            )
        else:
            # Default evening time
            optimal_times["evening"] = base_time.replace(
                hour=20, minute=0, second=0, microsecond=0
            )

        # Focus session times (based on productivity patterns)
        productivity_times = patterns.get("time_preferences", {}).get("productivity", [])
        if productivity_times:
            # Find most common productive hour
            from collections import Counter
            hour_counts = Counter(productivity_times)
            best_focus_hour = hour_counts.most_common(1)[0][0] if hour_counts else 10
            optimal_times["focus"] = base_time.replace(
                hour=best_focus_hour, minute=0, second=0, microsecond=0
            )
        else:
            optimal_times["focus"] = base_time.replace(
                hour=10, minute=0, second=0, microsecond=0
            )

        return optimal_times

    def _load_patterns(self) -> List[ActivityPattern]:
        """Load learned activity patterns from disk."""
        pattern_file = os.path.join(self.data_dir, "activity_patterns.json")
        if os.path.exists(pattern_file):
            with open(pattern_file, "r") as f:
                data = json.load(f)
                patterns = []
                for p in data:
                    last_occurrence = None
                    if p.get("last_occurrence"):
                        last_occurrence = datetime.fromisoformat(p["last_occurrence"])
                        last_occurrence = ensure_timezone_aware(last_occurrence)

                    patterns.append(ActivityPattern(
                        activity_type=p["activity_type"],
                        typical_start=time.fromisoformat(p["typical_start"]),
                        typical_duration=p["typical_duration"],
                        days_of_week=p["days_of_week"],
                        confidence=p["confidence"],
                        last_occurrence=last_occurrence
                    ))
                return patterns
        return []

    def _load_transition_times(self) -> Dict[TransitionType, List[time]]:
        """Load learned transition times."""
        transition_file = os.path.join(self.data_dir, "transitions.json")
        if os.path.exists(transition_file):
            with open(transition_file, "r") as f:
                data = json.load(f)
                transitions = {}
                for key, times in data.items():
                    transitions[TransitionType(key)] = [
                        time.fromisoformat(t) for t in times
                    ]
                return transitions
        return defaultdict(list)

    def analyze_patterns(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze behavioral patterns from recent data.

        Args:
            days: Number of days to look back

        Returns:
            Pattern analysis results
        """
        # Use timezone-aware datetime for cutoff
        cutoff = ensure_timezone_aware(datetime.now(timezone.utc)) - timedelta(days=days)

        # Get logs after cutoff
        log_nodes = self.graph.search_nodes(
            node_type="log",
            created_after=cutoff.isoformat(),
            limit=1000
        )

        # Analyze patterns
        pattern = {
            "total_logs": len(log_nodes),
            "days_analyzed": days,
            "daily_patterns": defaultdict(list),
            "weekly_patterns": defaultdict(list),
            "category_frequency": defaultdict(int),
            "time_preferences": defaultdict(list)
        }

        # Process each node
        for node in log_nodes:
            try:
                timestamp_str = node.get('created_at', '') or node.get('timestamp', '')

                if timestamp_str:
                    if isinstance(timestamp_str, str):
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        timestamp = timestamp_str

                    # Always ensure timezone awareness
                    timestamp = ensure_timezone_aware(timestamp)

                    # Now safe to use timestamp
                    hour = timestamp.hour
                    weekday = timestamp.weekday()

                    category = node.get('attrs', {}).get('category', 'unknown')
                    pattern["daily_patterns"][hour].append(category)
                    pattern["weekly_patterns"][weekday].append(category)
                    pattern["category_frequency"][category] += 1
                    pattern["time_preferences"][category].append(hour)

            except Exception as e:
                # Skip problematic nodes
                continue

        # Calculate optimal check-in times
        pattern["optimal_times"] = self._calculate_optimal_times(pattern)

        return pattern

    def _cluster_times(self, occurrences: List[Dict]) -> List[List[Dict]]:
        """Cluster activity occurrences by time of day."""
        # Simple clustering: group if within 2 hours
        clusters = []
        for occ in occurrences:
            placed = False
            for cluster in clusters:
                # Check if close to cluster average
                avg_minutes = statistics.mean([
                    t["time"].hour * 60 + t["time"].minute
                    for t in cluster
                ])
                occ_minutes = occ["time"].hour * 60 + occ["time"].minute

                if abs(occ_minutes - avg_minutes) < 120:  # Within 2 hours
                    cluster.append(occ)
                    placed = True
                    break

            if not placed:
                clusters.append([occ])

        return clusters

    def generate_daily_schedule(self, date: datetime) -> List[CheckInSchedule]:
        """
        Generate optimal check-in schedule for a given day.

        Args:
            date: The date to schedule for

        Returns:
            List of scheduled check-ins
        """
        # Ensure date is timezone-aware
        date = ensure_timezone_aware(date)
        schedule = []
        day_of_week = date.weekday()

        # Get relevant patterns for this day
        active_patterns = [
            p for p in self.activity_patterns
            if p.is_active_on_day(day_of_week)
            and p.confidence >= self.pattern_confidence_threshold
        ]

        # Schedule morning check-in based on wake pattern
        wake_time = self._predict_wake_time(date)
        if wake_time:
            # Create timezone-aware scheduled time
            scheduled_dt = ensure_timezone_aware(
                datetime.combine(date.date(), wake_time) + timedelta(minutes=15)
            )

            morning_check = CheckInSchedule(
                check_in_type=CheckInType.MORNING,
                scheduled_time=scheduled_dt,
                priority=0.9,
                context={"expected_topics": ["sleep", "mood", "energy", "planning"]}
            )
            schedule.append(morning_check)

        # Schedule periodic check-ins around transitions
        work_start = self._find_pattern_time(active_patterns, "work")
        if work_start:
            # Pre-work focus check
            focus_dt = ensure_timezone_aware(
                datetime.combine(date.date(), work_start) - timedelta(minutes=10)
            )

            focus_check = CheckInSchedule(
                check_in_type=CheckInType.FOCUS,
                scheduled_time=focus_dt,
                priority=0.7,
                context={"purpose": "pre_work_state"}
            )
            schedule.append(focus_check)

            # Mid-day energy check
            midday_dt = ensure_timezone_aware(
                datetime.combine(date.date(), work_start) + timedelta(hours=4)
            )

            midday_check = CheckInSchedule(
                check_in_type=CheckInType.PERIODIC,
                scheduled_time=midday_dt,
                priority=0.6,
                context={"expected_topics": ["energy", "focus", "nutrition"]}
            )
            schedule.append(midday_check)

        # Schedule evening reflection
        wind_down = self._predict_wind_down_time(date)
        if wind_down:
            evening_dt = ensure_timezone_aware(
                datetime.combine(date.date(), wind_down)
            )

            evening_check = CheckInSchedule(
                check_in_type=CheckInType.EVENING,
                scheduled_time=evening_dt,
                priority=0.8,
                context={"expected_topics": ["accomplishments", "social", "reflection"]}
            )
            schedule.append(evening_check)

        # Add medication reminders based on profile
        med_schedule = self._generate_medication_reminders(date)
        schedule.extend(med_schedule)

        # Sort by time and priority
        schedule.sort(key=lambda s: (ensure_timezone_aware(s.scheduled_time), -s.priority))

        self.current_schedule = schedule
        self._save_schedule(date, schedule)

        return schedule

    def _predict_wake_time(self, date: datetime) -> Optional[time]:
        """Predict wake time based on patterns."""
        # Look for sleep end patterns
        if TransitionType.WAKE_UP in self.transition_times:
            times = self.transition_times[TransitionType.WAKE_UP]
            if times:
                # Use median for robustness
                minutes = [t.hour * 60 + t.minute for t in times]
                median_minutes = statistics.median(minutes)
                return time(
                    hour=int(median_minutes // 60),
                    minute=int(median_minutes % 60)
                )

        # Default to profile routine if available
        routines = self.profile.get_facts_by_category("routines")
        for fact in routines:
            if "wake" in fact.get("fact", "").lower():
                # Try to extract time from fact
                # This is simplified - real implementation would be more robust
                return time(7, 0)  # Default morning time

        return time(7, 30)  # Fallback default

    def _predict_wind_down_time(self, date: datetime) -> Optional[time]:
        """Predict evening wind-down time."""
        if TransitionType.WIND_DOWN in self.transition_times:
            times = self.transition_times[TransitionType.WIND_DOWN]
            if times:
                minutes = [t.hour * 60 + t.minute for t in times]
                median_minutes = statistics.median(minutes)
                return time(
                    hour=int(median_minutes // 60),
                    minute=int(median_minutes % 60)
                )

        # Default based on sleep patterns
        return time(21, 0)  # 9 PM default

    def _find_pattern_time(self, patterns: List[ActivityPattern], activity_type: str) -> Optional[time]:
        """Find typical time for an activity type."""
        matching = [p for p in patterns if activity_type in p.activity_type.lower()]
        if matching:
            # Return highest confidence pattern
            best = max(matching, key=lambda p: p.confidence)
            return best.typical_start
        return None

    def _generate_medication_reminders(self, date: datetime) -> List[CheckInSchedule]:
        """Generate medication reminder check-ins."""
        # Ensure date is timezone-aware
        date = ensure_timezone_aware(date)
        reminders = []

        # Get medication facts from profile
        health_facts = self.profile.get_facts_by_category("health")
        for fact in health_facts:
            if "medication" in fact.get("fact", "").lower():
                # Parse medication schedule (simplified)
                # Real implementation would parse actual medication times

                # Example: L-theanine reminder
                if "theanine" in fact.get("fact", "").lower():
                    # Schedule for typical coffee times
                    coffee_times = self._find_coffee_times()
                    for coffee_time in coffee_times:
                        reminder_dt = ensure_timezone_aware(
                            datetime.combine(date.date(), coffee_time)
                        )

                        reminder = CheckInSchedule(
                            check_in_type=CheckInType.FOCUS,
                            scheduled_time=reminder_dt,
                            priority=0.5,
                            context={
                                "type": "medication_reminder",
                                "medication": "L-theanine",
                                "reason": "Take with coffee for focus"
                            }
                        )
                        reminders.append(reminder)

        return reminders

    def _find_coffee_times(self) -> List[time]:
        """Find typical coffee consumption times."""
        coffee_pattern = next(
            (p for p in self.activity_patterns if "coffee" in p.activity_type.lower()),
            None
        )
        if coffee_pattern:
            return [coffee_pattern.typical_start]
        return [time(8, 0), time(14, 0)]  # Default morning and afternoon

    def schedule_pomodoro(
        self,
        task: str,
        start_time: Optional[datetime] = None,
        duration: int = 25,
        break_duration: int = 5
    ) -> PomodoroSession:
        """
        Schedule a pomodoro work session.

        Args:
            task: Task description
            start_time: When to start (None for now)
            duration: Work duration in minutes
            break_duration: Break duration in minutes

        Returns:
            Scheduled pomodoro session
        """
        if start_time is None:
            start_time = ensure_timezone_aware(datetime.now(timezone.utc))
        else:
            start_time = ensure_timezone_aware(start_time)

        session = PomodoroSession(
            start_time=start_time,
            duration_minutes=duration,
            break_minutes=break_duration,
            task=task
        )

        self.pomodoro_sessions.append(session)

        # Log the start of the session
        self.logger.log({
            "category": "task",
            "value": f"Started pomodoro: {task}",
            "timestamp": start_time.isoformat(),
            "confidence": 1.0,
            "metadata": {
                "type": "pomodoro_start",
                "duration": duration,
                "task": task
            }
        })

        return session

    def complete_pomodoro(self, session: PomodoroSession, focus_score: float) -> None:
        """
        Mark a pomodoro session as complete.

        Args:
            session: The completed session
            focus_score: Focus level during session (0.0-1.0)
        """
        session.completed = True
        session.focus_score = focus_score

        # Log completion with timezone-aware timestamp
        self.logger.log({
            "category": "task",
            "value": f"Completed pomodoro: {session.task}",
            "timestamp": ensure_timezone_aware(datetime.now(timezone.utc)).isoformat(),
            "confidence": 1.0,
            "metadata": {
                "type": "pomodoro_complete",
                "duration": session.duration_minutes,
                "focus_score": focus_score,
                "task": session.task
            }
        })

        # Update patterns based on successful session
        self._update_work_patterns(session)

    def _update_work_patterns(self, session: PomodoroSession) -> None:
        """Update work patterns based on completed session."""
        if session.focus_score and session.focus_score > 0.7:
            # High focus session - remember this time as good for work
            session_time = ensure_timezone_aware(session.start_time)
            work_time = session_time.time()

            # Find or create work pattern
            work_pattern = next(
                (p for p in self.activity_patterns if p.activity_type == "focused_work"),
                None
            )

            if work_pattern:
                # Update pattern with exponential moving average
                old_minutes = work_pattern.typical_start.hour * 60 + work_pattern.typical_start.minute
                new_minutes = work_time.hour * 60 + work_time.minute
                avg_minutes = old_minutes * (1 - self.learning_rate) + new_minutes * self.learning_rate

                work_pattern.typical_start = time(
                    hour=int(avg_minutes // 60),
                    minute=int(avg_minutes % 60)
                )
                work_pattern.confidence = min(work_pattern.confidence + 0.05, 1.0)
                work_pattern.last_occurrence = session_time
            else:
                # Create new pattern
                self.activity_patterns.append(ActivityPattern(
                    activity_type="focused_work",
                    typical_start=work_time,
                    typical_duration=session.duration_minutes,
                    days_of_week=[session_time.weekday()],
                    confidence=0.5,
                    last_occurrence=session_time
                ))

            self._save_patterns()

    def update_transition(self, transition_type: TransitionType, timestamp: datetime) -> None:
        """
        Update learned transition times.

        Args:
            transition_type: Type of transition
            timestamp: When it occurred
        """
        timestamp = ensure_timezone_aware(timestamp)
        transition_time = timestamp.time()
        self.transition_times[transition_type].append(transition_time)

        # Keep only recent transitions (last 30 occurrences)
        if len(self.transition_times[transition_type]) > 30:
            self.transition_times[transition_type] = self.transition_times[transition_type][-30:]

        self._save_transitions()

    def get_next_check_in(self) -> Optional[CheckInSchedule]:
        """Get the next scheduled check-in."""
        now = ensure_timezone_aware(datetime.now(timezone.utc))
        upcoming = [
            s for s in self.current_schedule
            if ensure_timezone_aware(s.scheduled_time) > now and not s.completed and not s.skipped
        ]

        if upcoming:
            return min(upcoming, key=lambda s: ensure_timezone_aware(s.scheduled_time))
        return None

    def complete_check_in(self, schedule: CheckInSchedule) -> None:
        """Mark a check-in as completed."""
        schedule.completed = True
        schedule.actual_time = ensure_timezone_aware(datetime.now(timezone.utc))

        # Learn from timing difference
        if schedule.scheduled_time and schedule.actual_time:
            scheduled = ensure_timezone_aware(schedule.scheduled_time)
            actual = ensure_timezone_aware(schedule.actual_time)
            diff_minutes = (actual - scheduled).seconds / 60

            # If consistently early/late, adjust future schedules
            # This would be more sophisticated in production
            if abs(diff_minutes) > 30:
                self._adjust_future_schedules(schedule.check_in_type, diff_minutes)

    def _adjust_future_schedules(self, check_in_type: CheckInType, offset_minutes: float) -> None:
        """Adjust future schedules based on actual completion times."""
        # Simple learning: shift future schedules slightly
        adjustment = offset_minutes * self.learning_rate

        for schedule in self.current_schedule:
            if schedule.check_in_type == check_in_type and not schedule.completed:
                schedule.scheduled_time = ensure_timezone_aware(schedule.scheduled_time) + timedelta(minutes=adjustment)

    def _save_patterns(self) -> None:
        """Save activity patterns to disk."""
        pattern_file = os.path.join(self.data_dir, "activity_patterns.json")
        data = []
        for p in self.activity_patterns:
            data.append({
                "activity_type": p.activity_type,
                "typical_start": p.typical_start.isoformat(),
                "typical_duration": p.typical_duration,
                "days_of_week": p.days_of_week,
                "confidence": p.confidence,
                "last_occurrence": ensure_timezone_aware(p.last_occurrence).isoformat() if p.last_occurrence else None
            })

        with open(pattern_file, "w") as f:
            json.dump(data, f, indent=2)

    def _save_transitions(self) -> None:
        """Save transition times to disk."""
        transition_file = os.path.join(self.data_dir, "transitions.json")
        data = {}
        for key, times in self.transition_times.items():
            data[key.value] = [t.isoformat() for t in times]

        with open(transition_file, "w") as f:
            json.dump(data, f, indent=2)

    def _save_schedule(self, date: datetime, schedule: List[CheckInSchedule]) -> None:
        """Save schedule to disk."""
        date = ensure_timezone_aware(date)
        schedule_file = os.path.join(
            self.data_dir,
            f"schedule_{date.strftime('%Y%m%d')}.json"
        )

        data = [s.to_dict() for s in schedule]
        with open(schedule_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_daily_summary(self) -> Dict[str, Any]:
        """Get summary of today's scheduled activities."""
        today = ensure_timezone_aware(datetime.now(timezone.utc)).date()

        completed = [s for s in self.current_schedule if s.completed]
        pending = [s for s in self.current_schedule if not s.completed and not s.skipped]

        # Calculate adherence
        total_scheduled = len(self.current_schedule)
        adherence = len(completed) / total_scheduled if total_scheduled > 0 else 0

        # Get pomodoro stats
        today_pomodoros = [
            p for p in self.pomodoro_sessions
            if ensure_timezone_aware(p.start_time).date() == today
        ]
        completed_pomodoros = [p for p in today_pomodoros if p.completed]
        avg_focus = statistics.mean([p.focus_score for p in completed_pomodoros if p.focus_score]) \
            if completed_pomodoros else 0

        return {
            "date": today.isoformat(),
            "scheduled_check_ins": total_scheduled,
            "completed_check_ins": len(completed),
            "pending_check_ins": len(pending),
            "adherence_rate": adherence,
            "pomodoros_completed": len(completed_pomodoros),
            "average_focus": avg_focus,
            "next_check_in": self.get_next_check_in().to_dict() if self.get_next_check_in() else None,
            "patterns_active": len([p for p in self.activity_patterns if p.confidence > 0.7])
        }


# Example usage and integration
if __name__ == "__main__":
    # Initialize components
    from assistant.logger.unified import UnifiedLogger
    from assistant.graph.store import GraphStore
    from assistant.logger.temporal_personal_profile import TemporalPersonalProfile

    logger = UnifiedLogger(data_dir="data/tracking")
    graph = GraphStore(data_dir="data/graph")
    profile = TemporalPersonalProfile()

    # Create scheduler
    scheduler = AdaptiveScheduler(logger, graph, profile)

    # Analyze patterns from last 30 days
    patterns = scheduler.analyze_patterns(days=30)
    print(f"Discovered {len(patterns)} activity patterns")

    # Generate today's schedule
    today = ensure_timezone_aware(datetime.now(timezone.utc))
    schedule = scheduler.generate_daily_schedule(today)
    print(f"\nToday's Schedule ({today.strftime('%A, %B %d')}):")
    for check_in in schedule:
        print(f"  {ensure_timezone_aware(check_in.scheduled_time).strftime('%H:%M')} - {check_in.check_in_type.value}")

    # Start a pomodoro
    session = scheduler.schedule_pomodoro("Review behavioral patterns code", duration=25)
    print(f"\nPomodoro started: {session.task}")

    # Get summary
    summary = scheduler.get_daily_summary()
    print(f"\nDaily Summary:")
    print(f"  Adherence: {summary['adherence_rate']:.1%}")
    print(f"  Patterns Active: {summary['patterns_active']}")