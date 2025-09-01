"""
Integration Layer for Smart Scheduler with Notifications
========================================================

Connects the adaptive scheduler to the conversational logger and adds
email notifications for check-ins and reminders.
"""

import os
import smtplib
import schedule
import time
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass
import json
from assistant.behavioral_engine.schedulers.adaptive_scheduler import PomodoroSession

from assistant.conversational_logger import ConversationalLogger, CheckInType
from assistant.logger.unified import UnifiedLogger
from assistant.graph.store import GraphStore


@dataclass
class NotificationConfig:
    """Configuration for notifications."""
    email_enabled: bool = True
    email_address: Optional[str] = None
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None  # Use app password for Gmail
    
    pushover_enabled: bool = False
    pushover_token: Optional[str] = None
    pushover_user: Optional[str] = None
    
    desktop_enabled: bool = True
    
    @classmethod
    def from_env(cls) -> "NotificationConfig":
        """Load configuration from environment variables."""
        return cls(
            email_enabled=os.getenv("ENABLE_EMAIL", "true").lower() == "true",
            email_address=os.getenv("NOTIFICATION_EMAIL"),
            smtp_username=os.getenv("SMTP_USERNAME"),
            smtp_password=os.getenv("SMTP_PASSWORD"),
            pushover_token=os.getenv("PUSHOVER_TOKEN"),
            pushover_user=os.getenv("PUSHOVER_USER"),
            desktop_enabled=os.getenv("ENABLE_DESKTOP", "true").lower() == "true"
        )


class NotificationManager:
    """Manages sending notifications through various channels."""
    
    def __init__(self, config: NotificationConfig):
        """Initialize notification manager."""
        self.config = config
        
    def send_notification(
        self,
        title: str,
        message: str,
        priority: str = "normal",
        context: Optional[Dict] = None
    ) -> bool:
        """
        Send notification through configured channels.
        
        Args:
            title: Notification title
            message: Notification body
            priority: Priority level (low, normal, high)
            context: Additional context for rich notifications
            
        Returns:
            True if at least one notification was sent successfully
        """
        success = False
        
        if self.config.email_enabled and self.config.email_address:
            try:
                self._send_email(title, message, context)
                success = True
            except Exception as e:
                print(f"Email notification failed: {e}")
        
        if self.config.desktop_enabled:
            try:
                self._send_desktop(title, message)
                success = True
            except Exception as e:
                print(f"Desktop notification failed: {e}")
        
        if self.config.pushover_enabled:
            try:
                self._send_pushover(title, message, priority)
                success = True
            except Exception as e:
                print(f"Pushover notification failed: {e}")
        
        return success
    
    def _send_email(self, subject: str, body: str, context: Optional[Dict] = None) -> None:
        """Send email notification."""
        if not all([self.config.smtp_username, self.config.smtp_password]):
            raise ValueError("SMTP credentials not configured")
        
        msg = MIMEMultipart()
        msg['From'] = self.config.smtp_username
        msg['To'] = self.config.email_address
        msg['Subject'] = f"🤖 {subject}"
        
        # Format body with context if available
        if context:
            body += "\n\n---\n"
            for key, value in context.items():
                body += f"{key}: {value}\n"
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
            server.starttls()
            server.login(self.config.smtp_username, self.config.smtp_password)
            server.send_message(msg)
    
    def _send_desktop(self, title: str, message: str) -> None:
        """Send desktop notification."""
        try:
            # Try plyer for cross-platform notifications
            from plyer import notification
            notification.notify(
                title=title,
                message=message,
                app_icon=None,
                timeout=10
            )
        except ImportError:
            # Fallback to OS-specific commands
            import platform
            if platform.system() == "Darwin":  # macOS
                os.system(f'osascript -e \'display notification "{message}" with title "{title}"\'')
            elif platform.system() == "Linux":
                os.system(f'notify-send "{title}" "{message}"')
            elif platform.system() == "Windows":
                # Windows notifications require more complex implementation
                print(f"Desktop notification: {title} - {message}")
    
    def _send_pushover(self, title: str, message: str, priority: str) -> None:
        """Send Pushover notification."""
        if not all([self.config.pushover_token, self.config.pushover_user]):
            raise ValueError("Pushover credentials not configured")
        
        import requests
        
        priority_map = {"low": -1, "normal": 0, "high": 1}
        
        response = requests.post("https://api.pushover.net/1/messages.json", data={
            "token": self.config.pushover_token,
            "user": self.config.pushover_user,
            "title": title,
            "message": message,
            "priority": priority_map.get(priority, 0)
        })
        
        response.raise_for_status()


class ScheduledCheckInOrchestrator:
    """
    Orchestrates scheduled check-ins with the conversational logger
    and manages notifications.
    """
    
    def __init__(
        self,
        scheduler,  # AdaptiveScheduler instance
        conv_logger: ConversationalLogger,
        notifier: NotificationManager,
        data_dir: str = "data/orchestrator"
    ):
        """Initialize the orchestrator."""
        self.scheduler = scheduler
        self.conv_logger = conv_logger
        self.notifier = notifier
        self.data_dir = data_dir
        
        os.makedirs(data_dir, exist_ok=True)
        
        # Track active sessions
        self.active_session = None
        self.pending_notifications = []
        
        # Start background scheduler thread
        self.scheduler_thread = None
        self.running = False
    
    def start(self) -> None:
        """Start the background scheduler."""
        if self.running:
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        print("Scheduler started. Running in background...")
    
    def stop(self) -> None:
        """Stop the background scheduler."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        print("Scheduler stopped.")
    
    def _run_scheduler(self) -> None:
        """Background thread that checks for scheduled events."""
        while self.running:
            try:
                # Check for upcoming check-ins
                next_check_in = self.scheduler.get_next_check_in()
                
                if next_check_in:
                    time_until = (next_check_in.scheduled_time - datetime.now(timezone.utc)).seconds
                    
                    # Send reminder 5 minutes before
                    if time_until <= 300 and time_until > 240:  # 5 minutes
                        self._send_check_in_reminder(next_check_in)
                    
                    # Send notification at scheduled time
                    elif time_until <= 60:  # Within 1 minute
                        self._trigger_check_in(next_check_in)
                
                # Check for pomodoro timers
                self._check_pomodoro_timers()
                
                # Check for medication reminders
                self._check_medication_reminders()
                
                # Sleep for 30 seconds before next check
                time.sleep(30)
                
            except Exception as e:
                print(f"Scheduler error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _send_check_in_reminder(self, check_in_schedule) -> None:
        """Send reminder notification for upcoming check-in."""
        # Avoid duplicate notifications
        notification_key = f"reminder_{check_in_schedule.scheduled_time.isoformat()}"
        if notification_key in self.pending_notifications:
            return
        
        self.pending_notifications.append(notification_key)
        
        title = f"Check-in in 5 minutes: {check_in_schedule.check_in_type.value}"
        message = self._format_check_in_message(check_in_schedule)
        
        self.notifier.send_notification(
            title=title,
            message=message,
            priority="normal",
            context=check_in_schedule.context
        )
    
    def _trigger_check_in(self, check_in_schedule) -> None:
        """Trigger a scheduled check-in."""
        notification_key = f"trigger_{check_in_schedule.scheduled_time.isoformat()}"
        if notification_key in self.pending_notifications:
            return
        
        self.pending_notifications.append(notification_key)
        
        # Send notification
        title = f"Time for {check_in_schedule.check_in_type.value} check-in"
        message = "Ready to check in? Reply 'yes' to start or 'skip' to postpone."
        
        self.notifier.send_notification(
            title=title,
            message=message,
            priority="normal" if check_in_schedule.priority < 0.8 else "high",
            context=check_in_schedule.context
        )
        
        # Mark as pending response
        self._save_pending_check_in(check_in_schedule)
    
    def _format_check_in_message(self, check_in_schedule) -> str:
        """Format check-in message based on type and context."""
        messages = {
            CheckInType.MORNING: "Good morning! Time to check in on sleep and plan your day.",
            CheckInType.PERIODIC: "Quick status check - how's your energy and focus?",
            CheckInType.EVENING: "Evening reflection time - let's review the day.",
            CheckInType.FOCUS: "Quick focus check - how's your concentration?"
        }
        
        base_message = messages.get(
            check_in_schedule.check_in_type,
            "Time for a check-in!"
        )
        
        # Add context-specific information
        if check_in_schedule.context.get("expected_topics"):
            topics = ", ".join(check_in_schedule.context["expected_topics"])
            base_message += f"\n\nTopics: {topics}"
        
        return base_message
    
    def _check_pomodoro_timers(self) -> None:
        """Check for pomodoro session ends."""
        now = datetime.now(timezone.utc)
        
        for session in self.scheduler.pomodoro_sessions:
            if not session.completed:
                # Check if session should end
                if now >= session.end_time:
                    # Send break notification
                    self.notifier.send_notification(
                        title="Pomodoro Complete! 🍅",
                        message=f"Great work on: {session.task}\nTime for a {session.break_minutes} minute break.",
                        priority="high"
                    )
                    
                    # Prompt for focus score
                    self._save_pending_pomodoro_review(session)
                
                # 5 minute warning
                elif (session.end_time - now).seconds <= 300 and (session.end_time - now).seconds > 240:
                    notification_key = f"pomo_warning_{session.start_time.isoformat()}"
                    if notification_key not in self.pending_notifications:
                        self.pending_notifications.append(notification_key)
                        self.notifier.send_notification(
                            title="5 minutes remaining",
                            message=f"Wrap up: {session.task}",
                            priority="low"
                        )
    
    def _check_medication_reminders(self) -> None:
        """Check for medication reminders."""
        now = datetime.now(timezone.utc)
        current_time = now.time()
        
        # Check if any medication reminders are due
        for check_in in self.scheduler.current_schedule:
            if (check_in.context.get("type") == "medication_reminder" 
                and not check_in.completed
                and not check_in.skipped):
                
                # Check if it's time
                scheduled_time = check_in.scheduled_time.time()
                time_diff = abs(
                    (current_time.hour * 60 + current_time.minute) -
                    (scheduled_time.hour * 60 + scheduled_time.minute)
                )
                
                if time_diff <= 5:  # Within 5 minutes
                    notification_key = f"med_{check_in.scheduled_time.isoformat()}"
                    if notification_key not in self.pending_notifications:
                        self.pending_notifications.append(notification_key)
                        
                        medication = check_in.context.get("medication", "medication")
                        reason = check_in.context.get("reason", "")
                        
                        self.notifier.send_notification(
                            title=f"Medication Reminder: {medication}",
                            message=reason or f"Time to take {medication}",
                            priority="normal"
                        )
                        
                        check_in.completed = True
    
    def start_check_in_from_notification(self, check_in_type: CheckInType) -> Dict:
        """Start a check-in session triggered by notification response."""
        # Start the check-in
        session_info = self.conv_logger.start_check_in(check_in_type)
        self.active_session = session_info
        
        # Find and mark the scheduled check-in as in progress
        for check_in in self.scheduler.current_schedule:
            if check_in.check_in_type == check_in_type and not check_in.completed:
                check_in.actual_time = datetime.now(timezone.utc)
                break
        
        return session_info
    
    def process_email_response(self, email_body: str) -> Dict:
        """
        Process email responses to notifications.
        
        Args:
            email_body: The email body text
            
        Returns:
            Response to send back
        """
        # Simple command parsing
        body_lower = email_body.lower().strip()
        
        if body_lower in ["yes", "start", "ok"]:
            # Start pending check-in
            next_check_in = self.scheduler.get_next_check_in()
            if next_check_in:
                return self.start_check_in_from_notification(next_check_in.check_in_type)
            else:
                return {"message": "No pending check-ins"}
        
        elif body_lower in ["skip", "later", "postpone"]:
            # Skip current check-in
            next_check_in = self.scheduler.get_next_check_in()
            if next_check_in:
                next_check_in.skipped = True
                return {"message": "Check-in skipped. I'll check with you later."}
        
        elif self.active_session:
            # Process as conversation response
            response = self.conv_logger.process_response(email_body)
            
            # Send follow-up email if not complete
            if not response.get("is_complete"):
                self.notifier.send_notification(
                    title="Check-in Response",
                    message=response["message"],
                    context={"session_time": response["session_time"]}
                )
            else:
                # Send summary
                self.notifier.send_notification(
                    title="Check-in Complete",
                    message=f"Great job! Logged {response['summary']['entries_logged']} entries.",
                    context=response["summary"]
                )
                self.active_session = None
            
            return response
        
        else:
            # Quick log
            result = self.conv_logger.quick_log(email_body)
            return {
                "message": f"Logged {result['logged_count']} entries",
                "entries": result["entries"]
            }
    
    def _save_pending_check_in(self, check_in_schedule) -> None:
        """Save pending check-in to disk."""
        pending_file = os.path.join(self.data_dir, "pending_check_ins.json")
        
        try:
            with open(pending_file, "r") as f:
                pending = json.load(f)
        except FileNotFoundError:
            pending = []
        
        pending.append(check_in_schedule.to_dict())
        
        with open(pending_file, "w") as f:
            json.dump(pending, f, indent=2)
    
    def _save_pending_pomodoro_review(self, session: PomodoroSession) -> None:
        """Save pending pomodoro review."""
        review_file = os.path.join(self.data_dir, "pending_pomodoro_reviews.json")
        
        try:
            with open(review_file, "r") as f:
                reviews = json.load(f)
        except FileNotFoundError:
            reviews = []
        
        reviews.append({
            "start_time": session.start_time.isoformat(),
            "task": session.task,
            "duration": session.duration_minutes
        })
        
        with open(review_file, "w") as f:
            json.dump(reviews, f, indent=2)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for a dashboard view."""
        summary = self.scheduler.get_daily_summary()
        
        # Add notification status
        summary["notifications_enabled"] = {
            "email": self.notifier.config.email_enabled,
            "desktop": self.notifier.config.desktop_enabled,
            "pushover": self.notifier.config.pushover_enabled
        }
        
        # Add active session info
        summary["active_session"] = self.active_session
        
        # Add upcoming schedule
        upcoming = []
        for check_in in self.scheduler.current_schedule:
            if not check_in.completed and not check_in.skipped:
                upcoming.append({
                    "time": check_in.scheduled_time.strftime("%H:%M"),
                    "type": check_in.check_in_type.value,
                    "priority": check_in.priority
                })
        summary["upcoming_check_ins"] = upcoming[:5]  # Next 5
        
        return summary


# Example usage
if __name__ == "__main__":
    from assistant.logger.unified import UnifiedLogger
    from assistant.graph.store import GraphStore
    from assistant.logger.temporal_personal_profile import TemporalPersonalProfile
    from app.openai_client import OpenAIClient
    
    # Initialize components
    logger = UnifiedLogger(data_dir="data/tracking")
    graph = GraphStore(data_dir="data/graph")
    profile = TemporalPersonalProfile()
    openai_client = OpenAIClient()
    
    # Create conversational logger
    conv_logger = ConversationalLogger(logger, openai_client)
    
    # Create scheduler (import from previous artifact)
    # Update this import
    from assistant.behavioral_engine.schedulers.adaptive_scheduler import AdaptiveScheduler, PomodoroSession

    scheduler = AdaptiveScheduler(logger, graph, profile)
    
    # Configure notifications
    notif_config = NotificationConfig.from_env()
    notifier = NotificationManager(notif_config)
    
    # Create orchestrator
    orchestrator = ScheduledCheckInOrchestrator(
        scheduler=scheduler,
        conv_logger=conv_logger,
        notifier=notifier
    )
    
    # Analyze patterns and generate schedule
    scheduler.analyze_patterns(days=30)
    today_schedule = scheduler.generate_daily_schedule(datetime.now(timezone.utc))
    
    print(f"Generated {len(today_schedule)} check-ins for today")
    
    # Start the orchestrator
    orchestrator.start()
    
    # Example: Start a pomodoro
    session = scheduler.schedule_pomodoro(
        task="Implement notification system",
        duration=25
    )
    
    print(f"Pomodoro started: {session.task}")
    print("Orchestrator running in background. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(60)
            # Print status every minute
            dashboard = orchestrator.get_dashboard_data()
            print(f"Status - Adherence: {dashboard['adherence_rate']:.1%}, "
                  f"Next: {dashboard.get('next_check_in', {}).get('type', 'None')}")
    except KeyboardInterrupt:
        orchestrator.stop()
        print("Stopped.")
