"""
RPG Gamification Layer for Behavioral Tracking
===============================================

Adds game mechanics like XP, skills, levels, and achievements
to make behavioral tracking more engaging.
"""

from __future__ import annotations
import json
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

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



class SkillCategory(Enum):
    """Categories of skills that can be leveled."""
    # Physical
    FITNESS = "fitness"
    NUTRITION = "nutrition"
    SLEEP = "sleep"
    
    # Mental
    FOCUS = "focus"
    LEARNING = "learning"
    CREATIVITY = "creativity"
    
    # Social
    COMMUNICATION = "communication"
    RELATIONSHIPS = "relationships"
    
    # Productivity
    ORGANIZATION = "organization"
    EXECUTION = "execution"
    
    # Wellness
    MINDFULNESS = "mindfulness"
    SELF_CARE = "self_care"


@dataclass
class Skill:
    """Represents a skill that can be leveled up."""
    name: str
    category: SkillCategory
    level: int = 1
    xp: float = 0.0
    total_xp: float = 0.0
    
    @property
    def xp_for_next_level(self) -> float:
        """XP required for next level (exponential curve)."""
        return 100 * (1.5 ** (self.level - 1))
    
    @property
    def progress_to_next_level(self) -> float:
        """Progress percentage to next level."""
        return (self.xp / self.xp_for_next_level) * 100
    
    def add_xp(self, amount: float) -> bool:
        """
        Add XP to skill.
        
        Returns:
            True if leveled up
        """
        self.xp += amount
        self.total_xp += amount
        
        leveled_up = False
        while self.xp >= self.xp_for_next_level:
            self.xp -= self.xp_for_next_level
            self.level += 1
            leveled_up = True
        
        return leveled_up


@dataclass
class Achievement:
    """Represents an achievement/badge."""
    id: str
    name: str
    description: str
    icon: str  # Emoji or icon identifier
    category: str
    points: int = 10
    unlocked: bool = False
    unlocked_at: Optional[datetime] = None
    progress: float = 0.0  # For progressive achievements
    target: float = 1.0
    
    @property
    def is_complete(self) -> bool:
        """Check if achievement is complete."""
        return self.progress >= self.target
    
    def update_progress(self, value: float) -> bool:
        """
        Update achievement progress.
        
        Returns:
            True if newly unlocked
        """
        if self.unlocked:
            return False
        
        self.progress = min(value, self.target)
        
        if self.is_complete and not self.unlocked:
            self.unlocked = True
            self.unlocked_at = datetime.now(timezone.utc)
            return True
        
        return False


@dataclass
class DailyQuest:
    """Daily quest/challenge."""
    id: str
    name: str
    description: str
    xp_reward: float
    skill_rewards: Dict[str, float]  # skill_name -> xp
    requirements: List[Dict[str, Any]]
    completed: bool = False
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    def check_completion(self, user_data: Dict) -> bool:
        """Check if quest requirements are met."""
        for req in self.requirements:
            req_type = req["type"]
            
            if req_type == "log_count":
                category = req["category"]
                target = req["count"]
                actual = user_data.get(f"{category}_count", 0)
                if actual < target:
                    return False
            
            elif req_type == "value_threshold":
                category = req["category"]
                threshold = req["threshold"]
                comparison = req.get("comparison", ">=")
                actual = user_data.get(f"{category}_value", 0)
                
                if comparison == ">=" and actual < threshold:
                    return False
                elif comparison == "<=" and actual > threshold:
                    return False
            
            elif req_type == "streak":
                category = req["category"]
                days = req["days"]
                actual_streak = user_data.get(f"{category}_streak", 0)
                if actual_streak < days:
                    return False
        
        return True


@dataclass
class CharacterStats:
    """RPG-style character statistics."""
    level: int = 1
    total_xp: float = 0.0
    health: int = 100
    max_health: int = 100
    energy: int = 100
    max_energy: int = 100
    motivation: int = 100
    max_motivation: int = 100
    
    # Attributes (D&D style)
    strength: int = 10  # Physical capability
    intelligence: int = 10  # Learning and problem-solving
    wisdom: int = 10  # Mindfulness and judgment
    charisma: int = 10  # Social skills
    constitution: int = 10  # Resilience and health
    dexterity: int = 10  # Focus and precision
    
    @property
    def xp_for_next_level(self) -> float:
        """XP required for next character level."""
        return 500 * (1.2 ** (self.level - 1))
    
    def add_xp(self, amount: float) -> bool:
        """Add XP and check for level up."""
        self.total_xp += amount
        
        xp_needed = self.xp_for_next_level
        current_level_xp = sum(500 * (1.2 ** i) for i in range(self.level - 1))
        
        if self.total_xp >= current_level_xp + xp_needed:
            self.level += 1
            # Increase max stats on level up
            self.max_health += 5
            self.max_energy += 5
            self.max_motivation += 5
            return True
        
        return False


class RPGSystem:
    """
    Main RPG gamification system that tracks progress,
    awards XP, and manages game mechanics.
    """
    
    def __init__(
        self,
        logger: UnifiedLogger,
        graph_store: GraphStore,
        data_dir: str = "data/rpg"
    ):
        """Initialize RPG system."""
        self.logger = logger
        self.graph = graph_store
        self.data_dir = data_dir
        
        os.makedirs(data_dir, exist_ok=True)
        
        # Load or initialize game state
        self.character = self._load_character()
        self.skills = self._load_skills()
        self.achievements = self._load_achievements()
        self.daily_quests = []
        self.quest_history = self._load_quest_history()
        
        # XP multipliers for different activities
        self.xp_multipliers = {
            "morning_checkin": 1.5,  # Bonus for consistency
            "exercise": 2.0,  # Higher reward for physical activity
            "meditation": 1.8,  # Mindfulness bonus
            "pomodoro_complete": 1.2,
            "daily_goal": 3.0,  # Big reward for achieving goals
        }
        
        # Initialize achievements if empty
        if not self.achievements:
            self.achievements = self._create_default_achievements()
    
    def _load_character(self) -> CharacterStats:
        """Load character stats from disk."""
        char_file = os.path.join(self.data_dir, "character.json")
        if os.path.exists(char_file):
            with open(char_file, "r") as f:
                data = json.load(f)
                return CharacterStats(**data)
        return CharacterStats()
    
    def _load_skills(self) -> Dict[str, Skill]:
        """Load skills from disk."""
        skills_file = os.path.join(self.data_dir, "skills.json")
        if os.path.exists(skills_file):
            with open(skills_file, "r") as f:
                data = json.load(f)
                skills = {}
                for name, skill_data in data.items():
                    skills[name] = Skill(
                        name=name,
                        category=SkillCategory(skill_data["category"]),
                        level=skill_data["level"],
                        xp=skill_data["xp"],
                        total_xp=skill_data["total_xp"]
                    )
                return skills
        
        # Initialize default skills
        return self._create_default_skills()
    
    def _create_default_skills(self) -> Dict[str, Skill]:
        """Create default skill set."""
        skills = {
            # Physical
            "Running": Skill("Running", SkillCategory.FITNESS),
            "Strength Training": Skill("Strength Training", SkillCategory.FITNESS),
            "Healthy Eating": Skill("Healthy Eating", SkillCategory.NUTRITION),
            "Sleep Hygiene": Skill("Sleep Hygiene", SkillCategory.SLEEP),
            
            # Mental
            "Deep Focus": Skill("Deep Focus", SkillCategory.FOCUS),
            "Speed Reading": Skill("Speed Reading", SkillCategory.LEARNING),
            "Creative Writing": Skill("Creative Writing", SkillCategory.CREATIVITY),
            
            # Social
            "Active Listening": Skill("Active Listening", SkillCategory.COMMUNICATION),
            "Networking": Skill("Networking", SkillCategory.RELATIONSHIPS),
            
            # Productivity
            "Task Management": Skill("Task Management", SkillCategory.ORGANIZATION),
            "Time Boxing": Skill("Time Boxing", SkillCategory.EXECUTION),
            
            # Wellness
            "Meditation": Skill("Meditation", SkillCategory.MINDFULNESS),
            "Emotional Regulation": Skill("Emotional Regulation", SkillCategory.SELF_CARE),
        }
        
        return skills
    
    def _load_achievements(self) -> List[Achievement]:
        """Load achievements from disk."""
        achievements_file = os.path.join(self.data_dir, "achievements.json")
        if os.path.exists(achievements_file):
            with open(achievements_file, "r") as f:
                data = json.load(f)
                achievements = []
                for ach_data in data:
                    ach = Achievement(**ach_data)
                    if ach_data.get("unlocked_at"):
                        ach.unlocked_at = ensure_timezone_aware(datetime.fromisoformat(ach_data["unlocked_at"]))
                    achievements.append(ach)
                return achievements
        return []
    
    def _create_default_achievements(self) -> List[Achievement]:
        """Create default achievement set."""
        achievements = [
            # Streak achievements
            Achievement(
                id="week_warrior",
                name="Week Warrior",
                description="Complete check-ins for 7 days straight",
                icon="🗓️",
                category="consistency",
                points=25,
                target=7
            ),
            Achievement(
                id="monthly_master",
                name="Monthly Master",
                description="Complete check-ins for 30 days straight",
                icon="📅",
                category="consistency",
                points=100,
                target=30
            ),
            
            # Activity achievements
            Achievement(
                id="marathon_training",
                name="Marathon in Training",
                description="Log 100km of running",
                icon="🏃",
                category="fitness",
                points=50,
                target=100
            ),
            Achievement(
                id="bookworm",
                name="Bookworm",
                description="Log 50 hours of reading",
                icon="📚",
                category="learning",
                points=30,
                target=50
            ),
            Achievement(
                id="zen_master",
                name="Zen Master",
                description="Meditate for 30 days",
                icon="🧘",
                category="mindfulness",
                points=40,
                target=30
            ),
            
            # Productivity achievements
            Achievement(
                id="pomodoro_pro",
                name="Pomodoro Pro",
                description="Complete 100 pomodoro sessions",
                icon="🍅",
                category="productivity",
                points=35,
                target=100
            ),
            Achievement(
                id="task_crusher",
                name="Task Crusher",
                description="Complete 500 tasks",
                icon="✅",
                category="productivity",
                points=45,
                target=500
            ),
            
            # Level achievements
            Achievement(
                id="level_10",
                name="Double Digits",
                description="Reach character level 10",
                icon="🔟",
                category="progression",
                points=50,
                target=10
            ),
            
            # Special achievements
            Achievement(
                id="night_owl",
                name="Night Owl",
                description="Complete 10 late-night check-ins (after 11 PM)",
                icon="🦉",
                category="special",
                points=20,
                target=10
            ),
            Achievement(
                id="early_bird",
                name="Early Bird",
                description="Complete 10 early morning check-ins (before 6 AM)",
                icon="🐦",
                category="special",
                points=20,
                target=10
            ),
        ]
        
        return achievements
    
    def _load_quest_history(self) -> List[DailyQuest]:
        """Load completed quest history."""
        history_file = os.path.join(self.data_dir, "quest_history.json")
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                data = json.load(f)
                quests = []
                for q_data in data:
                    quest = DailyQuest(**q_data)
                    if q_data.get("completed_at"):
                        quest.completed_at = ensure_timezone_aware(datetime.fromisoformat(q_data["completed_at"]))
                    if q_data.get("expires_at"):
                        quest.expires_at = ensure_timezone_aware(datetime.fromisoformat(q_data["expires_at"]))
                    quests.append(quest)
                return quests
        return []
    
    def process_activity(
        self,
        category: str,
        value: Any,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process an activity and award XP/achievements.
        
        Args:
            category: Activity category
            value: Activity value
            metadata: Additional context
            
        Returns:
            Results including XP gained, skills leveled, achievements unlocked
        """
        results = {
            "xp_gained": 0,
            "skills_xp": {},
            "skills_leveled": [],
            "achievements_unlocked": [],
            "character_leveled": False,
            "quest_progress": []
        }
        
        # Calculate base XP
        base_xp = self._calculate_base_xp(category, value, metadata)
        
        # Apply multipliers
        multiplier = 1.0
        if metadata:
            for key, mult in self.xp_multipliers.items():
                if key in metadata:
                    multiplier *= mult
        
        total_xp = base_xp * multiplier
        results["xp_gained"] = total_xp
        
        # Award character XP
        if self.character.add_xp(total_xp):
            results["character_leveled"] = True
        
        # Award skill XP
        skill_allocations = self._determine_skill_allocation(category, metadata)
        for skill_name, allocation in skill_allocations.items():
            if skill_name in self.skills:
                skill_xp = total_xp * allocation
                skill = self.skills[skill_name]
                
                if skill.add_xp(skill_xp):
                    results["skills_leveled"].append(skill_name)
                
                results["skills_xp"][skill_name] = skill_xp
        
        # Check achievements
        unlocked = self._check_achievements(category, value, metadata)
        results["achievements_unlocked"] = unlocked
        
        # Check daily quests
        quest_progress = self._update_quest_progress(category, value, metadata)
        results["quest_progress"] = quest_progress
        
        # Update character stats based on activity
        self._update_character_stats(category, value)
        
        # Save state
        self._save_state()
        
        return results
    
    def _calculate_base_xp(self, category: str, value: Any, metadata: Optional[Dict]) -> float:
        """Calculate base XP for an activity."""
        xp_table = {
            "exercise": 15,
            "nutrition": 5,
            "sleep": 10,
            "mood": 3,
            "task": 8,
            "focus": 5,
            "social": 7,
            "meditation": 12,
            "reading": 10,
            "writing": 10,
        }
        
        base = xp_table.get(category, 5)
        
        # Adjust based on value
        if isinstance(value, (int, float)):
            # Scale XP with value (e.g., exercise duration)
            if category == "exercise":
                base *= (value / 30)  # Normalized to 30 minutes
            elif category == "focus":
                base *= (value / 5)  # Normalized to 5/10 focus score
        
        # Bonus for high-quality activities
        if metadata and metadata.get("quality", 0) > 0.8:
            base *= 1.2
        
        return base
    
    def _determine_skill_allocation(
        self, 
        category: str, 
        metadata: Optional[Dict]
    ) -> Dict[str, float]:
        """Determine which skills get XP from an activity."""
        allocations = {}
        
        skill_map = {
            "exercise": {
                "running": ["Running"],
                "strength": ["Strength Training"],
                "yoga": ["Meditation"],
                "default": ["Running", "Strength Training"]
            },
            "nutrition": ["Healthy Eating"],
            "sleep": ["Sleep Hygiene"],
            "focus": ["Deep Focus", "Time Boxing"],
            "task": ["Task Management", "Time Boxing"],
            "social": ["Active Listening", "Networking"],
            "meditation": ["Meditation", "Emotional Regulation"],
            "reading": ["Speed Reading"],
            "writing": ["Creative Writing"],
        }
        
        if category == "exercise" and metadata:
            exercise_type = metadata.get("type", "default")
            skills = skill_map["exercise"].get(exercise_type, skill_map["exercise"]["default"])
        else:
            skills = skill_map.get(category, [])
        
        # Distribute XP among relevant skills
        if skills:
            allocation_per_skill = 1.0 / len(skills)
            for skill in skills:
                allocations[skill] = allocation_per_skill
        
        return allocations
    
    def _update_character_stats(self, category: str, value: Any) -> None:
        """Update character stats based on activity."""
        # Energy changes
        if category == "exercise":
            self.character.energy = max(0, self.character.energy - 20)
            self.character.health = min(self.character.max_health, self.character.health + 5)
        elif category == "sleep" and isinstance(value, (int, float)) and value >= 7:
            self.character.energy = self.character.max_energy
        elif category == "nutrition":
            self.character.energy = min(self.character.max_energy, self.character.energy + 10)
        
        # Motivation changes
        if category == "task":
            self.character.motivation = min(self.character.max_motivation, self.character.motivation + 5)
        elif category == "meditation":
            self.character.motivation = min(self.character.max_motivation, self.character.motivation + 10)
    
    def _check_achievements(
        self, 
        category: str, 
        value: Any, 
        metadata: Optional[Dict]
    ) -> List[str]:
        """Check and unlock achievements."""
        unlocked = []
        
        for achievement in self.achievements:
            if achievement.unlocked:
                continue
            
            # Check category-specific achievements
            if achievement.id == "marathon_training" and category == "exercise":
                if metadata and metadata.get("type") == "running":
                    distance = metadata.get("distance_km", 0)
                    achievement.progress += distance
                    if achievement.update_progress(achievement.progress):
                        unlocked.append(achievement.name)
            
            elif achievement.id == "pomodoro_pro" and category == "task":
                if metadata and metadata.get("type") == "pomodoro_complete":
                    achievement.progress += 1
                    if achievement.update_progress(achievement.progress):
                        unlocked.append(achievement.name)
            
            elif achievement.id == "level_10":
                achievement.progress = self.character.level
                if achievement.update_progress(achievement.progress):
                    unlocked.append(achievement.name)
        
        return unlocked
    
    def _update_quest_progress(
        self, 
        category: str, 
        value: Any, 
        metadata: Optional[Dict]
    ) -> List[str]:
        """Update daily quest progress."""
        completed_quests = []
        
        for quest in self.daily_quests:
            if quest.completed:
                continue
            
            # Build user data for quest checking
            user_data = {
                f"{category}_count": 1,
                f"{category}_value": value if isinstance(value, (int, float)) else 1,
            }
            
            if quest.check_completion(user_data):
                quest.completed = True
                quest.completed_at = datetime.now(timezone.utc)
                
                # Award quest rewards
                self.character.add_xp(quest.xp_reward)
                for skill_name, skill_xp in quest.skill_rewards.items():
                    if skill_name in self.skills:
                        self.skills[skill_name].add_xp(skill_xp)
                
                completed_quests.append(quest.name)
        
        return completed_quests
    
    def generate_daily_quests(self) -> List[DailyQuest]:
        """Generate daily quests based on user patterns and level."""
        quests = []
        
        # Base quest templates scaled by level
        level_multiplier = 1 + (self.character.level * 0.1)
        
        quest_templates = [
            {
                "id": f"daily_checkin_{datetime.now(timezone.utc).date()}",
                "name": "Check-In Champion",
                "description": "Complete 3 check-ins today",
                "xp_reward": 50 * level_multiplier,
                "skill_rewards": {"Task Management": 20},
                "requirements": [
                    {"type": "log_count", "category": "checkin", "count": 3}
                ]
            },
            {
                "id": f"exercise_quest_{datetime.now(timezone.utc).date()}",
                "name": "Move Your Body",
                "description": "Log at least 30 minutes of exercise",
                "xp_reward": 75 * level_multiplier,
                "skill_rewards": {"Running": 30, "Strength Training": 20},
                "requirements": [
                    {"type": "value_threshold", "category": "exercise", "threshold": 30}
                ]
            },
            {
                "id": f"focus_quest_{datetime.now(timezone.utc).date()}",
                "name": "Deep Work",
                "description": "Complete 3 pomodoro sessions with focus > 7",
                "xp_reward": 60 * level_multiplier,
                "skill_rewards": {"Deep Focus": 40, "Time Boxing": 20},
                "requirements": [
                    {"type": "log_count", "category": "pomodoro", "count": 3},
                    {"type": "value_threshold", "category": "focus", "threshold": 7}
                ]
            },
            {
                "id": f"wellness_quest_{datetime.now(timezone.utc).date()}",
                "name": "Mind and Body",
                "description": "Meditate and log good sleep (>7 hours)",
                "xp_reward": 80 * level_multiplier,
                "skill_rewards": {"Meditation": 30, "Sleep Hygiene": 30},
                "requirements": [
                    {"type": "log_count", "category": "meditation", "count": 1},
                    {"type": "value_threshold", "category": "sleep", "threshold": 7}
                ]
            }
        ]
        
        # Select 3 random quests for the day
        import random
        selected = random.sample(quest_templates, min(3, len(quest_templates)))
        
        for template in selected:
            quest = DailyQuest(
                id=template["id"],
                name=template["name"],
                description=template["description"],
                xp_reward=template["xp_reward"],
                skill_rewards=template["skill_rewards"],
                requirements=template["requirements"],
                expires_at=datetime.now(timezone.utc).replace(
                    hour=23, minute=59, second=59
                )
            )
            quests.append(quest)
        
        self.daily_quests = quests
        return quests
    
    def get_character_summary(self) -> Dict[str, Any]:
        """Get comprehensive character summary."""
        return {
            "character": {
                "level": self.character.level,
                "total_xp": self.character.total_xp,
                "xp_to_next": self.character.xp_for_next_level,
                "health": f"{self.character.health}/{self.character.max_health}",
                "energy": f"{self.character.energy}/{self.character.max_energy}",
                "motivation": f"{self.character.motivation}/{self.character.max_motivation}",
                "attributes": {
                    "STR": self.character.strength,
                    "INT": self.character.intelligence,
                    "WIS": self.character.wisdom,
                    "CHA": self.character.charisma,
                    "CON": self.character.constitution,
                    "DEX": self.character.dexterity,
                }
            },
            "top_skills": sorted(
                [(s.name, s.level, s.progress_to_next_level) for s in self.skills.values()],
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "achievements": {
                "unlocked": len([a for a in self.achievements if a.unlocked]),
                "total": len(self.achievements),
                "points": sum(a.points for a in self.achievements if a.unlocked)
            },
            "daily_quests": [
                {
                    "name": q.name,
                    "description": q.description,
                    "xp_reward": q.xp_reward,
                    "completed": q.completed
                }
                for q in self.daily_quests
            ]
        }
    
    def _save_state(self) -> None:
        """Save all RPG state to disk."""
        # Save character
        char_file = os.path.join(self.data_dir, "character.json")
        with open(char_file, "w") as f:
            char_data = {
                "level": self.character.level,
                "total_xp": self.character.total_xp,
                "health": self.character.health,
                "max_health": self.character.max_health,
                "energy": self.character.energy,
                "max_energy": self.character.max_energy,
                "motivation": self.character.motivation,
                "max_motivation": self.character.max_motivation,
                "strength": self.character.strength,
                "intelligence": self.character.intelligence,
                "wisdom": self.character.wisdom,
                "charisma": self.character.charisma,
                "constitution": self.character.constitution,
                "dexterity": self.character.dexterity,
            }
            json.dump(char_data, f, indent=2)
        
        # Save skills
        skills_file = os.path.join(self.data_dir, "skills.json")
        with open(skills_file, "w") as f:
            skills_data = {}
            for name, skill in self.skills.items():
                skills_data[name] = {
                    "category": skill.category.value,
                    "level": skill.level,
                    "xp": skill.xp,
                    "total_xp": skill.total_xp
                }
            json.dump(skills_data, f, indent=2)
        
        # Save achievements
        achievements_file = os.path.join(self.data_dir, "achievements.json")
        with open(achievements_file, "w") as f:
            ach_data = []
            for ach in self.achievements:
                ach_dict = {
                    "id": ach.id,
                    "name": ach.name,
                    "description": ach.description,
                    "icon": ach.icon,
                    "category": ach.category,
                    "points": ach.points,
                    "unlocked": ach.unlocked,
                    "progress": ach.progress,
                    "target": ach.target
                }
                if ach.unlocked_at:
                    ach_dict["unlocked_at"] = ach.unlocked_at.isoformat()
                ach_data.append(ach_dict)
            json.dump(ach_data, f, indent=2)


# Example integration
if __name__ == "__main__":
    from assistant.logger.unified import UnifiedLogger
    from assistant.graph.store import GraphStore
    
    # Initialize
    logger = UnifiedLogger(data_dir="data/tracking")
    graph = GraphStore(data_dir="data/graph")
    rpg = RPGSystem(logger, graph)
    
    # Generate daily quests
    quests = rpg.generate_daily_quests()
    print("Today's Quests:")
    for quest in quests:
        print(f"  📜 {quest.name}: {quest.description} ({quest.xp_reward} XP)")
    
    # Process some activities
    print("\n--- Processing Activities ---")
    
    # Morning run
    result = rpg.process_activity(
        "exercise",
        30,  # 30 minutes
        {"type": "running", "distance_km": 5, "morning_checkin": True}
    )
    print(f"Morning Run: +{result['xp_gained']:.1f} XP")
    if result["skills_leveled"]:
        print(f"  Skills leveled up: {', '.join(result['skills_leveled'])}")
    
    # Complete pomodoro
    result = rpg.process_activity(
        "task",
        1,
        {"type": "pomodoro_complete", "focus_score": 8}
    )
    print(f"Pomodoro Complete: +{result['xp_gained']:.1f} XP")
    
    # Check character summary
    summary = rpg.get_character_summary()
    print(f"\n--- Character Summary ---")
    print(f"Level {summary['character']['level']} "
          f"({summary['character']['total_xp']:.0f} XP)")
    print(f"Health: {summary['character']['health']}")
    print(f"Energy: {summary['character']['energy']}")
    print(f"Top Skills:")
    for name, level, progress in summary['top_skills']:
        print(f"  {name}: Lvl {level} ({progress:.1f}% to next)")
    print(f"Achievements: {summary['achievements']['unlocked']}/{summary['achievements']['total']} "
          f"({summary['achievements']['points']} points)")
