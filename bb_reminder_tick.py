# bb_reminder_tick.py
import os
from pathlib import Path
from assistant.behavioral_engine.schedulers.schedule_bridge import ReminderDaemon, EmailNotifier
rem = ReminderDaemon(Path("data/planner/reminders.jsonl"), EmailNotifier(os.environ))
rem.tick()