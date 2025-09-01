"""
Behavioral Routine Builder with Task Analysis
==============================================

Build, track, and analyze multi-step behavioral routines like wind-down,
morning routines, or any habit stack you're developing.
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timedelta, timezone, time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# Use app's OpenAI client for API quirks handling
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
class RoutineStep:
    """A single step in a behavioral routine."""
    
    name: str
    description: str
    estimated_minutes: float
    category: str  # hygiene, medication, relaxation, etc.
    required: bool = True  # Is this step mandatory?
    
    # Tracking
    completed: bool = False
    completed_at: Optional[datetime] = None
    skipped: bool = False
    skip_reason: Optional[str] = None
    actual_minutes: Optional[float] = None
    
    # Dependencies
    prerequisites: List[str] = field(default_factory=list)  # Other steps that must come first
    triggers: List[str] = field(default_factory=list)  # What triggers this step
    
    # Completion criteria
    completion_check: Optional[str] = None  # Natural language description
    measurement: Optional[str] = None  # How to measure completion
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "description": self.description,
            "estimated_minutes": self.estimated_minutes,
            "category": self.category,
            "required": self.required,
            "completed": self.completed,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "actual_minutes": self.actual_minutes,
            "prerequisites": self.prerequisites,
            "triggers": self.triggers,
            "completion_check": self.completion_check,
            "measurement": self.measurement
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "RoutineStep":
        """Create from dictionary."""
        step = cls(
            name=data["name"],
            description=data["description"],
            estimated_minutes=data["estimated_minutes"],
            category=data["category"],
            required=data.get("required", True)
        )
        
        step.completed = data.get("completed", False)
        if data.get("completed_at"):
            step.completed_at = ensure_timezone_aware(datetime.fromisoformat(data["completed_at"]))
        step.skipped = data.get("skipped", False)
        step.skip_reason = data.get("skip_reason")
        step.actual_minutes = data.get("actual_minutes")
        step.prerequisites = data.get("prerequisites", [])
        step.triggers = data.get("triggers", [])
        step.completion_check = data.get("completion_check")
        step.measurement = data.get("measurement")
        
        return step


@dataclass
class BehavioralRoutine:
    """A complete behavioral routine with multiple steps."""
    
    name: str
    description: str
    routine_type: str  # morning, evening, wind-down, exercise, etc.
    steps: List[RoutineStep] = field(default_factory=list)
    
    # Timing
    typical_start_time: Optional[time] = None
    total_estimated_minutes: float = 0.0
    
    # Tracking
    executions: List[Dict] = field(default_factory=list)  # History of executions
    current_execution: Optional[Dict] = None
    
    # Analysis
    completion_rate: float = 0.0
    average_duration: float = 0.0
    common_skip_points: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    
    def add_step(
        self,
        name: str,
        description: str,
        estimated_minutes: float,
        category: str,
        **kwargs
    ) -> RoutineStep:
        """Add a step to the routine."""
        step = RoutineStep(
            name=name,
            description=description,
            estimated_minutes=estimated_minutes,
            category=category,
            **kwargs
        )
        self.steps.append(step)
        self.total_estimated_minutes = sum(s.estimated_minutes for s in self.steps)
        return step
    
    def start_execution(self) -> Dict:
        """Start executing the routine."""
        self.current_execution = {
            "started_at": datetime.now(timezone.utc),
            "steps_completed": [],
            "steps_skipped": [],
            "total_duration": 0
        }
        
        # Reset step states
        for step in self.steps:
            step.completed = False
            step.completed_at = None
            step.skipped = False
            step.skip_reason = None
            step.actual_minutes = None
        
        return self.current_execution
    
    def complete_step(self, step_name: str, actual_minutes: Optional[float] = None) -> bool:
        """Mark a step as completed."""
        for step in self.steps:
            if step.name == step_name:
                step.completed = True
                step.completed_at = datetime.now(timezone.utc)
                step.actual_minutes = actual_minutes or step.estimated_minutes
                
                if self.current_execution:
                    self.current_execution["steps_completed"].append(step_name)
                    self.current_execution["total_duration"] += step.actual_minutes
                
                return True
        return False
    
    def skip_step(self, step_name: str, reason: str = "") -> bool:
        """Skip a step with optional reason."""
        for step in self.steps:
            if step.name == step_name:
                step.skipped = True
                step.skip_reason = reason
                
                if self.current_execution:
                    self.current_execution["steps_skipped"].append({
                        "step": step_name,
                        "reason": reason
                    })
                
                return True
        return False
    
    def complete_execution(self) -> Dict:
        """Complete the current execution and analyze."""
        if not self.current_execution:
            return {"error": "No active execution"}
        
        self.current_execution["completed_at"] = datetime.now(timezone.utc)
        self.current_execution["duration_minutes"] = (
            self.current_execution["completed_at"] - self.current_execution["started_at"]
        ).total_seconds() / 60
        
        # Calculate completion percentage
        required_steps = [s for s in self.steps if s.required]
        completed_required = [s for s in required_steps if s.completed]
        self.current_execution["completion_percentage"] = (
            len(completed_required) / len(required_steps) * 100 
            if required_steps else 100
        )
        
        # Store execution
        self.executions.append(self.current_execution)
        
        # Update analytics
        self._update_analytics()
        
        result = self.current_execution
        self.current_execution = None
        
        return result
    
    def _update_analytics(self):
        """Update routine analytics based on execution history."""
        if not self.executions:
            return
        
        # Completion rate
        completed_executions = [
            e for e in self.executions 
            if e.get("completion_percentage", 0) >= 80
        ]
        self.completion_rate = len(completed_executions) / len(self.executions)
        
        # Average duration
        durations = [e.get("duration_minutes", 0) for e in self.executions if e.get("duration_minutes")]
        self.average_duration = sum(durations) / len(durations) if durations else 0
        
        # Common skip points
        skip_counts = defaultdict(int)
        for execution in self.executions:
            for skip in execution.get("steps_skipped", []):
                skip_counts[skip["step"]] += 1
        
        # Find steps skipped >30% of the time
        total_executions = len(self.executions)
        self.common_skip_points = [
            step for step, count in skip_counts.items()
            if count / total_executions > 0.3
        ]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "description": self.description,
            "routine_type": self.routine_type,
            "steps": [s.to_dict() for s in self.steps],
            "typical_start_time": self.typical_start_time.isoformat() if self.typical_start_time else None,
            "total_estimated_minutes": self.total_estimated_minutes,
            "executions": self.executions,
            "completion_rate": self.completion_rate,
            "average_duration": self.average_duration,
            "common_skip_points": self.common_skip_points,
            "optimization_suggestions": self.optimization_suggestions
        }


class RoutineBuilder:
    """
    System for building, tracking, and optimizing behavioral routines.
    """
    
    def __init__(
        self,
        logger: UnifiedLogger,
        graph_store: GraphStore,
        openai_client: OpenAIClient,  # Using app's client
        data_dir: str = "data/routines"
    ):
        """Initialize routine builder."""
        self.logger = logger
        self.graph = graph_store
        self.openai_client = openai_client
        self.data_dir = data_dir
        
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing routines
        self.routines = self._load_routines()
        
        # Active routine tracking
        self.active_routine = None
        self.routine_timers = {}
    
    def _load_routines(self) -> Dict[str, BehavioralRoutine]:
        """Load saved routines from disk."""
        routines = {}
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith("_routine.json"):
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, "r") as f:
                    data = json.load(f)
                    routine = BehavioralRoutine(
                        name=data["name"],
                        description=data["description"],
                        routine_type=data["routine_type"]
                    )
                    
                    # Load steps
                    for step_data in data.get("steps", []):
                        routine.steps.append(RoutineStep.from_dict(step_data))
                    
                    # Load other fields
                    if data.get("typical_start_time"):
                        routine.typical_start_time = time.fromisoformat(data["typical_start_time"])
                    routine.total_estimated_minutes = data.get("total_estimated_minutes", 0)
                    routine.executions = data.get("executions", [])
                    routine.completion_rate = data.get("completion_rate", 0)
                    routine.average_duration = data.get("average_duration", 0)
                    routine.common_skip_points = data.get("common_skip_points", [])
                    routine.optimization_suggestions = data.get("optimization_suggestions", [])
                    
                    routines[routine.name] = routine
        
        return routines
    
    def create_routine_from_description(self, description: str) -> BehavioralRoutine:
        """
        Create a routine from natural language description using OpenAI.
        
        Args:
            description: Natural language description of the routine
            
        Returns:
            Created routine
        """
        prompt = f"""Analyze this routine description and break it into specific steps:

{description}

Return a JSON structure with:
{{
    "name": "routine name",
    "description": "brief description",
    "routine_type": "morning|evening|wind-down|exercise|work|other",
    "steps": [
        {{
            "name": "step name",
            "description": "what to do",
            "estimated_minutes": number,
            "category": "hygiene|medication|relaxation|exercise|nutrition|preparation|other",
            "required": true/false,
            "prerequisites": ["previous step names"],
            "completion_check": "how to verify completion",
            "measurement": "what to measure"
        }}
    ]
}}

Be specific and actionable. Include all mentioned activities."""

        try:
            response = self.openai_client.chat(
                messages=[
                    {"role": "system", "content": "You are a behavioral routine analyst."},
                    {"role": "user", "content": prompt}
                ],
                model="gpt-4o-mini"
            )
            
            # Parse response
            content = response.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            
            data = json.loads(content)
            
            # Create routine
            routine = BehavioralRoutine(
                name=data["name"],
                description=data["description"],
                routine_type=data["routine_type"]
            )
            
            # Add steps
            for step_data in data["steps"]:
                routine.add_step(
                    name=step_data["name"],
                    description=step_data["description"],
                    estimated_minutes=step_data["estimated_minutes"],
                    category=step_data["category"],
                    required=step_data.get("required", True),
                    prerequisites=step_data.get("prerequisites", []),
                    completion_check=step_data.get("completion_check"),
                    measurement=step_data.get("measurement")
                )
            
            # Save routine
            self.routines[routine.name] = routine
            self._save_routine(routine)
            
            return routine
            
        except Exception as e:
            print(f"Error creating routine: {e}")
            # Return a default routine
            return self._create_default_wind_down_routine()
    
    def _create_default_wind_down_routine(self) -> BehavioralRoutine:
        """Create the default wind-down routine you mentioned."""
        routine = BehavioralRoutine(
            name="Evening Wind-Down",
            description="Complete evening wind-down routine for better sleep",
            routine_type="wind-down"
        )
        
        routine.add_step(
            name="Shower",
            description="Take a warm relaxing shower",
            estimated_minutes=10,
            category="hygiene",
            completion_check="Showered and dried off"
        )
        
        routine.add_step(
            name="Brush Teeth",
            description="Brush teeth for 2 minutes",
            estimated_minutes=2,
            category="hygiene",
            prerequisites=["Shower"],
            completion_check="Teeth brushed",
            measurement="2 minutes brushing"
        )
        
        routine.add_step(
            name="Shave",
            description="Shave if needed",
            estimated_minutes=5,
            category="hygiene",
            required=False,
            prerequisites=["Shower"],
            completion_check="Face shaved or skipped if not needed"
        )
        
        routine.add_step(
            name="Holy Basil Tea",
            description="Prepare and drink cup of holy basil tea",
            estimated_minutes=5,
            category="relaxation",
            prerequisites=["Brush Teeth"],
            completion_check="Tea prepared and started drinking"
        )
        
        routine.add_step(
            name="Night Medications",
            description="Take prescribed night medications",
            estimated_minutes=2,
            category="medication",
            prerequisites=["Holy Basil Tea"],
            completion_check="All night meds taken",
            measurement="List of meds taken"
        )
        
        routine.add_step(
            name="Wind-Down Activity",
            description="Journaling or reading for 15-30 minutes",
            estimated_minutes=20,
            category="relaxation",
            prerequisites=["Night Medications"],
            completion_check="Completed journaling or reading",
            measurement="Pages read or journal entries written"
        )
        
        routine.typical_start_time = time(21, 30)  # 9:30 PM
        
        return routine
    
    def start_routine(self, routine_name: str) -> Dict:
        """Start executing a routine."""
        if routine_name not in self.routines:
            return {"error": f"Routine '{routine_name}' not found"}
        
        routine = self.routines[routine_name]
        self.active_routine = routine
        
        # Start execution
        execution = routine.start_execution()
        
        # Log start
        self.logger.log({
            "category": "routine",
            "value": f"Started {routine_name}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "routine_name": routine_name,
                "estimated_minutes": routine.total_estimated_minutes,
                "steps": len(routine.steps)
            }
        })
        
        # Start timer
        self.routine_timers[routine_name] = datetime.now(timezone.utc)
        
        return {
            "routine": routine_name,
            "started_at": execution["started_at"].isoformat(),
            "steps": [s.name for s in routine.steps],
            "estimated_minutes": routine.total_estimated_minutes
        }
    
    def complete_step(self, step_name: str, notes: str = "") -> Dict:
        """Complete a step in the active routine."""
        if not self.active_routine:
            return {"error": "No active routine"}
        
        # Calculate actual duration
        actual_minutes = None
        if step_name in self.routine_timers:
            duration = (datetime.now(timezone.utc) - self.routine_timers[step_name]).total_seconds() / 60
            actual_minutes = round(duration, 1)
        
        # Complete the step
        success = self.active_routine.complete_step(step_name, actual_minutes)
        
        if success:
            # Log completion
            self.logger.log({
                "category": "routine_step",
                "value": f"Completed: {step_name}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "routine": self.active_routine.name,
                    "step": step_name,
                    "actual_minutes": actual_minutes,
                    "notes": notes
                }
            })
            
            # Get next step
            next_step = self._get_next_step()
            
            # Start timer for next step
            if next_step:
                self.routine_timers[next_step.name] = datetime.now(timezone.utc)
            
            return {
                "step_completed": step_name,
                "actual_minutes": actual_minutes,
                "next_step": next_step.name if next_step else None,
                "progress": self._get_routine_progress()
            }
        
        return {"error": f"Step '{step_name}' not found"}
    
    def skip_step(self, step_name: str, reason: str) -> Dict:
        """Skip a step with a reason."""
        if not self.active_routine:
            return {"error": "No active routine"}
        
        success = self.active_routine.skip_step(step_name, reason)
        
        if success:
            # Log skip
            self.logger.log({
                "category": "routine_step",
                "value": f"Skipped: {step_name}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "routine": self.active_routine.name,
                    "step": step_name,
                    "reason": reason
                }
            })
            
            # Get next step
            next_step = self._get_next_step()
            
            return {
                "step_skipped": step_name,
                "reason": reason,
                "next_step": next_step.name if next_step else None,
                "progress": self._get_routine_progress()
            }
        
        return {"error": f"Step '{step_name}' not found"}
    
    def complete_routine(self) -> Dict:
        """Complete the active routine and get analysis."""
        if not self.active_routine:
            return {"error": "No active routine"}
        
        # Complete execution
        result = self.active_routine.complete_execution()
        
        # Log completion
        self.logger.log({
            "category": "routine",
            "value": f"Completed {self.active_routine.name}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "routine_name": self.active_routine.name,
                "duration_minutes": result.get("duration_minutes"),
                "completion_percentage": result.get("completion_percentage"),
                "steps_completed": len(result.get("steps_completed", [])),
                "steps_skipped": len(result.get("steps_skipped", []))
            }
        })
        
        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions()
        self.active_routine.optimization_suggestions = suggestions
        
        # Save routine with updated analytics
        self._save_routine(self.active_routine)
        
        # Clear active routine
        routine_name = self.active_routine.name
        self.active_routine = None
        self.routine_timers.clear()
        
        return {
            "routine": routine_name,
            "result": result,
            "analytics": {
                "completion_rate": self.routines[routine_name].completion_rate,
                "average_duration": self.routines[routine_name].average_duration,
                "common_skip_points": self.routines[routine_name].common_skip_points
            },
            "suggestions": suggestions
        }
    
    def _get_next_step(self) -> Optional[RoutineStep]:
        """Get the next uncompleted step in the routine."""
        if not self.active_routine:
            return None
        
        for step in self.active_routine.steps:
            if not step.completed and not step.skipped:
                # Check prerequisites
                if step.prerequisites:
                    prereqs_met = all(
                        any(s.name == prereq and s.completed 
                            for s in self.active_routine.steps)
                        for prereq in step.prerequisites
                    )
                    if not prereqs_met:
                        continue
                
                return step
        
        return None
    
    def _get_routine_progress(self) -> Dict:
        """Get current progress of active routine."""
        if not self.active_routine:
            return {}
        
        total_steps = len(self.active_routine.steps)
        completed_steps = sum(1 for s in self.active_routine.steps if s.completed)
        skipped_steps = sum(1 for s in self.active_routine.steps if s.skipped)
        
        required_steps = [s for s in self.active_routine.steps if s.required]
        required_completed = sum(1 for s in required_steps if s.completed)
        
        return {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "skipped_steps": skipped_steps,
            "remaining_steps": total_steps - completed_steps - skipped_steps,
            "completion_percentage": (completed_steps / total_steps * 100) if total_steps > 0 else 0,
            "required_completion": (required_completed / len(required_steps) * 100) if required_steps else 100
        }
    
    def _generate_optimization_suggestions(self) -> List[str]:
        """Generate suggestions for optimizing the routine using AI."""
        if not self.active_routine:
            return []
        
        # Prepare execution data
        execution_summary = {
            "routine": self.active_routine.name,
            "total_executions": len(self.active_routine.executions),
            "completion_rate": self.active_routine.completion_rate,
            "average_duration": self.active_routine.average_duration,
            "estimated_duration": self.active_routine.total_estimated_minutes,
            "common_skip_points": self.active_routine.common_skip_points,
            "recent_executions": self.active_routine.executions[-5:] if self.active_routine.executions else []
        }
        
        prompt = f"""Analyze this routine execution data and suggest optimizations:

{json.dumps(execution_summary, indent=2)}

Provide 3-5 specific, actionable suggestions to improve:
1. Completion rate
2. Time efficiency
3. Habit formation

Return as JSON array of strings."""

        try:
            response = self.openai_client.chat(
                messages=[
                    {"role": "system", "content": "You are a behavioral optimization expert."},
                    {"role": "user", "content": prompt}
                ],
                model="gpt-4o-mini"
            )
            
            # Parse response
            content = response.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            
            suggestions = json.loads(content)
            return suggestions if isinstance(suggestions, list) else []
            
        except Exception as e:
            print(f"Error generating suggestions: {e}")
            return [
                "Consider reducing the number of steps if completion rate is low",
                "Try batching similar activities together",
                "Set up environmental triggers for routine initiation"
            ]
    
    def analyze_routine_effectiveness(self, routine_name: str) -> Dict:
        """Analyze effectiveness of a routine over time."""
        if routine_name not in self.routines:
            return {"error": f"Routine '{routine_name}' not found"}
        
        routine = self.routines[routine_name]
        
        if len(routine.executions) < 3:
            return {
                "routine": routine_name,
                "message": "Need at least 3 executions for analysis",
                "executions": len(routine.executions)
            }
        
        # Time-based analysis
        recent_executions = routine.executions[-10:]  # Last 10
        older_executions = routine.executions[:-10] if len(routine.executions) > 10 else []
        
        recent_completion = sum(
            1 for e in recent_executions 
            if e.get("completion_percentage", 0) >= 80
        ) / len(recent_executions) if recent_executions else 0
        
        older_completion = sum(
            1 for e in older_executions 
            if e.get("completion_percentage", 0) >= 80
        ) / len(older_executions) if older_executions else 0
        
        # Trend analysis
        if recent_completion > older_completion:
            trend = "improving"
        elif recent_completion < older_completion:
            trend = "declining"
        else:
            trend = "stable"
        
        # Step-level analysis
        step_completion_rates = {}
        for step in routine.steps:
            completed_count = sum(
                1 for e in routine.executions
                if step.name in e.get("steps_completed", [])
            )
            step_completion_rates[step.name] = (
                completed_count / len(routine.executions) * 100
                if routine.executions else 0
            )
        
        # Find problematic steps
        problematic_steps = [
            step for step, rate in step_completion_rates.items()
            if rate < 50
        ]
        
        return {
            "routine": routine_name,
            "total_executions": len(routine.executions),
            "overall_completion_rate": routine.completion_rate * 100,
            "recent_completion_rate": recent_completion * 100,
            "trend": trend,
            "average_duration": routine.average_duration,
            "estimated_duration": routine.total_estimated_minutes,
            "time_variance": abs(routine.average_duration - routine.total_estimated_minutes),
            "step_completion_rates": step_completion_rates,
            "problematic_steps": problematic_steps,
            "optimization_suggestions": routine.optimization_suggestions
        }
    
    def _save_routine(self, routine: BehavioralRoutine):
        """Save routine to disk."""
        filename = f"{routine.name.replace(' ', '_').lower()}_routine.json"
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(routine.to_dict(), f, indent=2, default=str)
    
    def get_all_routines(self) -> Dict[str, Dict]:
        """Get summary of all routines."""
        summaries = {}
        
        for name, routine in self.routines.items():
            summaries[name] = {
                "description": routine.description,
                "type": routine.routine_type,
                "steps": len(routine.steps),
                "estimated_minutes": routine.total_estimated_minutes,
                "executions": len(routine.executions),
                "completion_rate": routine.completion_rate * 100,
                "typical_start": routine.typical_start_time.isoformat() if routine.typical_start_time else None
            }
        
        return summaries


# Example usage
if __name__ == "__main__":
    from assistant.logger.unified import UnifiedLogger
    from assistant.graph.store import GraphStore
    from app.openai_client import OpenAIClient
    
    # Initialize
    logger = UnifiedLogger(data_dir="data/tracking")
    graph = GraphStore(data_dir="data/graph")
    openai_client = OpenAIClient()  # Using app's client with quirks handling
    
    builder = RoutineBuilder(logger, graph, openai_client)
    
    # Create routine from description
    description = """
    Wind-down routine: shower, brush teeth, shave if needed, 
    make a cup of holy basil tea, take night medications including melatonin,
    then do a wind-down activity like journaling or reading for 15-30 minutes
    """
    
    routine = builder.create_routine_from_description(description)
    print(f"Created routine: {routine.name}")
    print(f"Total time: {routine.total_estimated_minutes} minutes")
    print(f"Steps:")
    for step in routine.steps:
        req = "Required" if step.required else "Optional"
        print(f"  - {step.name} ({step.estimated_minutes} min) [{req}]")
    
    # Start the routine
    print("\n--- Starting Routine ---")
    result = builder.start_routine(routine.name)
    print(f"Started at: {result['started_at']}")
    
    # Simulate completing steps
    print("\n--- Executing Steps ---")
    
    # Complete shower
    builder.complete_step("Shower", "Nice hot shower")
    print("✓ Shower completed")
    
    # Complete teeth brushing
    builder.complete_step("Brush Teeth", "Used electric toothbrush")
    print("✓ Brush Teeth completed")
    
    # Skip shaving
    builder.skip_step("Shave", "Not needed today")
    print("⊘ Shave skipped")
    
    # Complete remaining steps
    builder.complete_step("Holy Basil Tea", "Made a strong cup")
    print("✓ Holy Basil Tea completed")
    
    builder.complete_step("Night Medications", "Took melatonin and magnesium")
    print("✓ Night Medications completed")
    
    builder.complete_step("Wind-Down Activity", "Journaled for 20 minutes")
    print("✓ Wind-Down Activity completed")
    
    # Complete routine
    print("\n--- Routine Complete ---")
    completion = builder.complete_routine()
    
    print(f"Duration: {completion['result']['duration_minutes']:.1f} minutes")
    print(f"Completion: {completion['result']['completion_percentage']:.0f}%")
    
    if completion['suggestions']:
        print("\nOptimization Suggestions:")
        for suggestion in completion['suggestions']:
            print(f"  • {suggestion}")
    
    # Analyze effectiveness
    print("\n--- Routine Analysis ---")
    analysis = builder.analyze_routine_effectiveness(routine.name)
    print(f"Trend: {analysis.get('trend', 'N/A')}")
    print(f"Time variance: {analysis.get('time_variance', 0):.1f} minutes")
