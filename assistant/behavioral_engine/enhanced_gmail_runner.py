#!/usr/bin/env python3
"""
BiggerBrother Complete System with Gmail OAuth Integration
==========================================================

Enhanced with "just chatting" mode, EnhancedLabelHarmonizer, and all critical fixes.
Run this to start the full system with email monitoring and notifications.

FIXED VERSION: Now properly uses dual scheduler support for full functionality.
"""
import collections
import os
import pathlib
import sys
import time
import threading
import logging
from datetime import datetime, timezone, timedelta
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from pathlib import Path
import pickle
import json
import re
from enum import Enum

# Add BiggerBrother to path
sys.path.insert(0, r'C:\BiggerBrother-minimal')

# Gmail OAuth imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import base64

# BiggerBrother imports
from assistant.behavioral_engine.core.complete_system import CompleteIntegratedSystem
from assistant.behavioral_engine.schedulers.adaptive_scheduler import AdaptiveScheduler
from assistant.behavioral_engine.schedulers.notification_integration import (
    NotificationConfig,
    NotificationManager,
    ScheduledCheckInOrchestrator
)
from assistant.conversational_logger import CheckInType
from assistant.logger.unified import UnifiedLogger
from assistant.graph.store import GraphStore
from assistant.logger.temporal_personal_profile import TemporalPersonalProfile
from app.openai_client import OpenAIClient
from assistant.behavioral_engine.schedulers.intelligent_scheduler import IntelligentSchedulingSystem
from assistant.behavioral_engine.schedulers.intelligent_scheduler import ScheduleIntentType

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]  # send+mark read; avoid /gmail.readonly if you modify
BASE = pathlib.Path("secrets")  # up to you
TOKEN = BASE / "token.json"
CLIENT = BASE / "credentials.json"
CHECKPOINT = pathlib.Path("run_state/processed_ids.json")
MAX_IDS = 5000  # rolling window

def load_processed():
    try:
        return collections.deque(json.loads(CHECKPOINT.read_text()), maxlen=MAX_IDS)
    except Exception:
        return collections.deque(maxlen=MAX_IDS)

def save_processed(deq):
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(list(deq)))


def gmail_service():
    BASE.mkdir(parents=True, exist_ok=True)
    creds = None
    if TOKEN.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT), SCOPES)
            creds = flow.run_local_server(port=0)
        # persist as JSON (not pickle)
        with open(TOKEN, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


class ConversationMode(Enum):
    """Different conversation modes for BiggerBrother."""
    CHECK_IN = "check_in"
    CHAT = "chat"  # Just chatting mode
    DETAILED = "detailed"  # Detailed activity logging
    QUICK_LOG = "quick_log"
    ROUTINE = "routine"


class GmailIntegration:
    """Gmail OAuth integration for BiggerBrother."""

    SCOPES = [
        'https://www.googleapis.com/auth/gmail.modify',
        'https://www.googleapis.com/auth/gmail.compose'
    ]

    def __init__(self, credentials_file='credentials.json', token_file='token.pickle'):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        self.email_address = None
        self._authenticate()

    def _authenticate(self):
        """Authenticate with Gmail using saved token or OAuth flow."""
        creds = None

        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.SCOPES
                )
                creds = flow.run_local_server(port=0)

            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)

        self.service = build('gmail', 'v1', credentials=creds)

        # Get email address
        profile = self.service.users().getProfile(userId='me').execute()
        self.email_address = profile['emailAddress']
        print(f"✅ Gmail connected: {self.email_address}")

    def get_unread_messages(self, query="is:unread subject:BiggerBrother"):
        """Get unread messages matching query with pagination (up to 10 pages)."""
        try:
            messages = []
            page_token = None
            pages = 0
            while True:
                results = self.service.users().messages().list(
                    userId='me', q=query, maxResults=50, pageToken=page_token
                ).execute()
                for msg in (results.get('messages') or []):
                    full_msg = self.service.users().messages().get(
                        userId='me', id=msg['id']
                    ).execute()
                    payload = full_msg.get('payload', {})
                    headers = {h['name']: h['value'] for h in payload.get('headers', [])}
                    body = self._extract_body(payload)
                    messages.append({
                        'id': msg['id'],
                        'threadId': full_msg.get('threadId'),
                        'from': headers.get('From', ''),
                        'subject': headers.get('Subject', ''),
                        'body': body
                    })
                page_token = results.get('nextPageToken')
                pages += 1
                if not page_token or pages >= 10:
                    break
            return messages
        except Exception as e:
            print(f"Error fetching messages: {e}")
            return []





    def _extract_body(self, payload):
        """Extract text body from message payload with HTML fallback."""
        body = ""
        # Multipart message
        if 'parts' in payload:
            text_plain = []
            text_html = []
            stack = list(payload.get('parts', []))
            # Walk nested multiparts
            while stack:
                part = stack.pop()
                if 'parts' in part:
                    stack.extend(part['parts'])
                    continue
                mime = part.get('mimeType', '')
                data = part.get('body', {}).get('data')
                if not data:
                    continue
                decoded = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                if mime == 'text/plain':
                    text_plain.append(decoded)
                elif mime == 'text/html':
                    text_html.append(decoded)
            if text_plain:
                body = "\n".join(text_plain)
            elif text_html:
                body = self._strip_html("\n".join(text_html))
        # Single-part body
        elif payload.get('body', {}).get('data'):
            try:
                body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
            except Exception:
                body = ""
        return (body or "").strip()
        
    def _strip_html(self, html: str) -> str:
        """Best-effort HTML → text without extra deps."""
        import re
        from html import unescape
        s = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", html)  # drop scripts/styles
        s = re.sub(r"(?i)<br\s*/?>", "\n", s)                      # <br> → newline
        s = re.sub(r"(?i)</p\s*>", "\n\n", s)                      # </p> → blank line
        s = re.sub(r"<[^>]+>", "", s)                               # strip tags
        s = unescape(s)
        # normalize whitespace
        return "\n".join(line.strip() for line in s.splitlines() if line.strip())

    def send_email(self, to=None, to_addr=None, subject="", body=None, text_body=None, html_body=None, html=None,
                   from_name=None, thread_id=None, service=None):
        """
        Backwards-compatible Gmail send. Accepts both (to, subject, body) and (to_addr, subject, text_body),
        and optionally a 'service'. Falls back to self.service.
        """
        service = service or self.service
        to_addr = to_addr or to
        text_body = text_body or body
        if html is not None and html_body is None:
            html_body = html
        if html_body:
            msg = MIMEMultipart("alternative")
            msg.attach(MIMEText(text_body or "", "plain", "utf-8"))
            msg.attach(MIMEText(html_body, "html", "utf-8"))
        else:
            msg = MIMEText(text_body or "", "plain", "utf-8")

        # optional friendly display name
        if from_name:
            msg["From"] = formataddr((str(Header(from_name, "utf-8")), "me"))
        else:
            msg["From"] = "me"  # Gmail API will resolve ‘me’ to the authenticated account

        msg["To"] = to_addr
        msg["Subject"] = str(Header(subject or "", "utf-8"))

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
        body = {"raw": raw}
        if thread_id:
            body["threadId"] = thread_id

        return service.users().messages().send(userId="me", body=body).execute()

    def mark_as_read(self, msg_id, service=None):
        """Mark message as read."""
        try:
            (service or self.service).users().messages().modify(
                userId='me',
                id=msg_id,
                body={'removeLabelIds': ['UNREAD']}
            ).execute()
        except Exception as e:
            print(f"Error marking as read: {e}")

    def _extract_message_data(self, full_msg):
        """Normalize a Gmail API message resource to a friendly dict."""
        payload = full_msg.get('payload', {})
        headers = {h['name']: h['value'] for h in payload.get('headers', [])}
        return {
            'id': full_msg.get('id'),
            'thread_id': full_msg.get('threadId'),
            'from': headers.get('From', ''),
            'subject': headers.get('Subject', ''),
            'body': self._extract_body(payload)
        }


class BiggerBrotherEmailSystem:
    """Complete BiggerBrother system with Gmail integration, chat mode, and EnhancedLabelHarmonizer."""

    def __init__(self, base_dir=None, use_config=True):
        """
        Initialize BiggerBrother Email System.

        Args:
            base_dir: Base directory for data (optional if use_config=True)
            use_config: Use centralized data_config.py paths (default: True)
        """
        self.running = False

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize Gmail
        self.gmail = GmailIntegration()

        # Initialize BiggerBrother with EnhancedLabelHarmonizer and dual schedulers
        self.logger.info("Initializing BiggerBrother system with EnhancedLabelHarmonizer and dual schedulers...")

        if use_config:
            # Use centralized configuration
            from data_config import ORCHESTRATOR_DIR

            self.system = CompleteIntegratedSystem(
                base_dir=None,  # Will use config
                use_config=True
            )
            self.base_dir = str(ORCHESTRATOR_DIR.parent)  # data/ directory
            orchestrator_dir = str(ORCHESTRATOR_DIR)
        else:
            # Traditional approach
            self.base_dir = base_dir or "data"
            self.system = CompleteIntegratedSystem(
                base_dir=self.base_dir,
                use_config=False
            )
            orchestrator_dir = f"{self.base_dir}/orchestrator"

        # Use BOTH schedulers from the system for different purposes
        self.adaptive_scheduler = self.system.adaptive_scheduler  # For patterns & scheduling
        self.context_scheduler = self.system.context_scheduler  # For message processing
        self.profile = self.system.profile

        # For backward compatibility, default to adaptive for Gmail runner methods
        self.scheduler = self.adaptive_scheduler

        # Setup notifications (using Gmail)
        self.notif_config = NotificationConfig(
            email_enabled=True,
            email_address=self.gmail.email_address,
            desktop_enabled=True
        )
        self.notifier = NotificationManager(self.notif_config)

        # Override email sending to use Gmail OAuth
        self._override_email_sending()

        # Create orchestrator with adaptive scheduler for scheduling features
        self.orchestrator = ScheduledCheckInOrchestrator(
            scheduler=self.adaptive_scheduler,  # Uses adaptive for scheduling
            conv_logger=None,
            notifier=self.notifier,
            data_dir=orchestrator_dir  # Uses configured path
        )

        # Track active conversations and their modes
        self.active_conversations = {}
        self.processed_emails = set()

        # Chat mode context tracking
        self.chat_contexts = {}

        # Initialize Intelligent Scheduling System
        self.scheduling_system = IntelligentSchedulingSystem(
            openai_client=self.system.openai_client,
            complete_system=None,  # Will be set next
            base_dir=self.base_dir
        )
        # Set references to avoid circular imports
        self.scheduling_system.set_complete_system(self.system)
        self.scheduling_system.email_system = self

        # Email monitoring thread
        self.email_thread = None

    def _override_email_sending(self):
        """Override the notification manager to use Gmail OAuth."""
        original_send = self.notifier._send_email

        def gmail_send(subject, body, context=None):
            return self.gmail.send_email(
                to=self.gmail.email_address,
                subject=f"🤖 {subject}",
                body=body
            )

        self.notifier._send_email = gmail_send

    def start(self):
        """Start all services."""
        if self.running:
            return

        self.running = True

        # Use adaptive_scheduler for pattern analysis and scheduling
        self.logger.info("Analyzing behavioral patterns...")
        self.adaptive_scheduler.analyze_patterns(days=30)

        today_schedule = self.adaptive_scheduler.generate_daily_schedule(datetime.now(timezone.utc))
        self.logger.info(f"Generated {len(today_schedule)} check-ins for today")

        # Start orchestrator
        self.orchestrator.start()

        # Start email monitoring
        self.email_thread = threading.Thread(target=self._monitor_emails, daemon=True)
        self.email_thread.start()

        self._print_status()

    def _monitor_emails(self):
        """Background thread to monitor emails with exponential backoff."""
        self.logger.info("📧 Email monitoring started")
        sleep_base = 15
        backoff = sleep_base
        max_backoff = 300
        while self.running:
            try:
                messages = self.gmail.get_unread_messages()
                for msg in messages:
                    if msg['id'] not in self.processed_emails:
                        self._process_email(msg)
                        self.processed_emails.add(msg['id'])
                backoff = sleep_base
                time.sleep(backoff)
            except Exception as e:
                self.logger.error(f"Email monitoring error: {e}")
                backoff = min(max_backoff, backoff * 2)
                time.sleep(backoff)


    def _get_conversation_mode(self, from_addr):
        """Get the current conversation mode for a user."""
        if from_addr in self.active_conversations:
            return self.active_conversations[from_addr].get('mode', ConversationMode.QUICK_LOG)
        return ConversationMode.QUICK_LOG

    def _process_email(self, msg):
        """Process an incoming email with mode awareness."""
        self.logger.info(f"📨 Processing email: {msg['subject']}")
        import re
        from_email = re.search(r'<(.+?)>', msg['from'])
        from_addr = from_email.group(1) if from_email else msg['from']
        body = msg['body'].strip()

        response = None
        current_mode = self._get_conversation_mode(from_addr)

        # Check for scheduling session confirmations
        if 'Session:' in body:
            # Extract session ID
            import re
            session_match = re.search(r'Session:\s*([^\s]+)', body)
            if session_match:
                session_id = session_match.group(1)
                # Check if this is a pending confirmation
                for full_id in self.scheduling_system.pending_confirmations:
                    if full_id.startswith(session_id):
                        result = self.scheduling_system.confirm_schedule(full_id, body)
                        response = result.get('message', result.get('response', 'Schedule updated'))
                        break

        # 1) Try intelligent scheduling on every inbound first
        if not response:
            try:
                sched = self.scheduling_system.process_message(body, source='email')
                if sched.get('scheduling', {}).get('intent_detected'):
                    response = sched['scheduling']['response']
            except Exception as e:
                self.logger.error(f"Scheduling parse error: {e}")

        # 2) If no response yet, continue with normal processing
        if not response:

            # Check for mode-specific commands first
            if current_mode != ConversationMode.QUICK_LOG:
                # Handle ongoing conversation
                if re.search(r'\b(done|exit|bye|stop)\b', body, re.IGNORECASE):
                    response = self._end_conversation(from_addr)
                else:
                    response = self._continue_conversation(from_addr, body, current_mode)

            # Check for new conversation starters
            elif re.search(r'\b(chat|talk|conversation)\b', body, re.IGNORECASE):
                response = self._start_chat_mode(from_addr)

            elif re.search(r'\b(checkin|check-in|start)\b', body, re.IGNORECASE):
                response = self._start_checkin_mode(from_addr)

            elif re.search(r'\b(detailed|activity|logging)\b', body, re.IGNORECASE):
                response = self._start_detailed_mode(from_addr)

            elif re.search(r'\b(help|commands?)\b', body, re.IGNORECASE):
                response = self._get_help_text()

            elif re.search(r'\b(status|summary|report)\b', body, re.IGNORECASE):
                response = self._get_status_summary()

            elif re.search(r'\b(skip|later|postpone|no)\b', body, re.IGNORECASE):
                response = "Check-in skipped. I'll check with you later."

            elif match := re.search(r'\bpomodoro\s+(.+)', body, re.IGNORECASE):
                # Use adaptive_scheduler for pomodoro
                task = match.group(1)
                session = self.adaptive_scheduler.schedule_pomodoro(task)
                response = f"🍅 Pomodoro started for: {task}\nEnds at {session.end_time.strftime('%H:%M')}"

            # Default: quick log
            else:
                result = self.system.process_message_with_context(body)
                response = self._format_quick_log_response(result)

            # Send response if needed
            if response:
                self.gmail.send_email(
                    to=from_addr,
                    subject=f"Re: {msg['subject']}",
                    body=response,
                    thread_id=msg['threadId']
                )

            # Mark as read
            self.gmail.mark_as_read(msg['id'])

    def _start_chat_mode(self, from_addr):
        """Start a 'just chatting' conversation mode."""
        # Initialize chat context
        self.chat_contexts[from_addr] = {
            'messages': [],
            'start_time': datetime.now(timezone.utc),
            'topics_discussed': set()
        }

        # Track conversation mode
        self.active_conversations[from_addr] = {
            'mode': ConversationMode.CHAT,
            'start_time': datetime.now(timezone.utc)
        }

        greeting = """💬 Chat mode started!

I'm here for an open conversation. We can explore topics in depth, reflect on patterns, or just talk about whatever's on your mind.

I'll still capture important information for your logs, but the focus is on having a natural, flowing conversation.

What would you like to talk about?

(Reply 'done' when you want to end the chat)"""

        return greeting

    def _start_checkin_mode(self, from_addr):
        """Start a structured check-in using context scheduler."""
        # Determine check-in type
        hour = datetime.now(timezone.utc).hour
        if 5 <= hour < 10:
            check_in_type = CheckInType.MORNING
        elif 17 <= hour < 22:
            check_in_type = CheckInType.EVENING
        else:
            check_in_type = CheckInType.PERIODIC

        # Start check-in through CompleteIntegratedSystem (uses context scheduler internally)
        session = self.system.start_check_in_with_logs(check_in_type)

        # Track the conversation mode
        self.active_conversations[from_addr] = {
            'mode': ConversationMode.CHECK_IN,
            'start_time': datetime.now(timezone.utc),
            'session_id': session['session_id']
        }

        return f"""✅ Check-in started!

{session['greeting']}

Reply to continue the conversation.
(Type 'done' when finished)"""

    def _start_detailed_mode(self, from_addr):
        """Start detailed activity logging mode."""
        session = self.system.start_detailed_activity_logging()

        self.active_conversations[from_addr] = {
            'mode': ConversationMode.DETAILED,
            'start_time': datetime.now(timezone.utc),
            'session': session
        }

        return f"""📝 Detailed Activity Logging Mode

{session['greeting']}

Describe your activities in detail and I'll extract and categorize everything.

(Type 'done' when finished)"""

    def _continue_conversation(self, from_addr, message, mode):
        """Continue a conversation based on its mode."""
        if mode == ConversationMode.CHAT:
            return self._continue_chat(from_addr, message)
        elif mode == ConversationMode.CHECK_IN:
            return self._continue_checkin(from_addr, message)
        elif mode == ConversationMode.DETAILED:
            return self._continue_detailed(from_addr, message)
        else:
            return self._format_quick_log_response(
                self.system.process_message_with_context(message)
            )

    def _continue_chat(self, from_addr, message):
        """Continue a 'just chatting' conversation with label awareness."""
        # Store message in context
        if from_addr in self.chat_contexts:
            self.chat_contexts[from_addr]['messages'].append({
                'role': 'user',
                'content': message,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        # Scheduling intent intercept
        try:
            sched = self.scheduling_system.process_message(message, source='chat')
            if sched.get('scheduling', {}).get('intent_detected'):
                return sched['scheduling']['response']
        except Exception:
            pass
        # Process with context awareness (uses context scheduler internally)
        result = self.system.process_message_with_context(message)

        # Extract label insights if available
        label_context = ""
        if result.get('label_insights'):
            insights = result['label_insights']
            topics = insights.get('topic', {})
            if topics.get('unique_concepts', 0) > 0:
                label_context = f"\n[Unique topic detected]"

        # Build contextual response for chat mode
        response = result.get('response', '')

        # In chat mode, we can be more conversational
        if not response:
            # Generate a more conversational response using GPT-4
            context_messages = self.chat_contexts[from_addr]['messages'][-5:] if from_addr in self.chat_contexts else []

            chat_prompt = f"""You're having a relaxed, in-depth conversation. 
Previous context: {json.dumps(context_messages, indent=2)}
{label_context}

User said: {message}

Respond naturally and explore the topic deeply. Ask thoughtful follow-up questions when appropriate."""

            try:
                response = self.system.openai_client.chat(
                    messages=[
                        {"role": "system", "content": chat_prompt},
                        {"role": "user", "content": message}
                    ],
                    model="gpt-4o"
                )
            except:
                response = "I'm listening. Tell me more about that."

        # Store assistant response
        if from_addr in self.chat_contexts:
            self.chat_contexts[from_addr]['messages'].append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        # Add subtle indicators of what was logged
        footer = ""
        if result.get('labels_generated', 0) > 0:
            footer = f"\n\n[{result['labels_generated']} topics noted]"
            if result.get('label_insights', {}).get('topic', {}).get('unique_concepts', 0) > 0:
                footer += " [unique concept detected]"

        return response + footer

    def _continue_checkin(self, from_addr, message):
        """Continue a check-in conversation (uses context scheduler internally)."""
        try:
            sched = self.scheduling_system.process_message(message, source='checkin')
            if sched.get('scheduling', {}).get('intent_detected'):
                return sched['scheduling']['response']
        except Exception:
            pass
        result = self.system.process_message_with_context(message)

        response = result.get('response', 'Got it!')

        # Add metadata
        metadata = []
        if result.get('extracted', []):
            metadata.append(f"Logged {len(result['extracted'])} items")
        if result.get('labels_generated', 0) > 0:
            metadata.append(f"{result['labels_generated']} labels")
        if result.get('label_insights'):
            insights = result['label_insights']
            unique_count = sum(v.get('unique_concepts', 0) for v in insights.values())
            if unique_count > 0:
                metadata.append(f"{unique_count} unique concepts")

        if metadata:
            response += f"\n\n[{', '.join(metadata)}]"

        return response

    def _continue_detailed(self, from_addr, message):
        """Continue detailed activity logging."""
        result = self.system.process_message_with_context(message)

        response = result.get('response', 'Noted!')

        if result.get('logged_to'):
            response += f"\n\n[Logged to: {', '.join(result['logged_to'])}]"

        return response

    def _end_conversation(self, from_addr):
        """End any active conversation."""
        mode = self._get_conversation_mode(from_addr)

        # Generate summary based on mode
        if mode == ConversationMode.CHAT:
            # Summarize chat conversation
            duration = datetime.now(timezone.utc) - self.active_conversations[from_addr]['start_time']
            minutes = int(duration.total_seconds() / 60)

            summary = f"""💬 Chat ended after {minutes} minutes.

Thanks for the conversation! I've captured the important points for your records."""

            # Clear chat context
            if from_addr in self.chat_contexts:
                del self.chat_contexts[from_addr]

        elif mode == ConversationMode.CHECK_IN:
            self.system.active_check_in = None
            summary = "✅ Check-in complete! Great job staying engaged."

        elif mode == ConversationMode.DETAILED:
            if hasattr(self.system, 'activity_logger'):
                session_summary = self.system.activity_logger.complete_session()
                summary = f"📝 Session complete: {session_summary['activities_extracted']} activities logged"
            else:
                summary = "📝 Detailed logging session complete."
            self.system.active_detailed_session = None

        else:
            summary = "Conversation ended."

        # Clear conversation tracking
        if from_addr in self.active_conversations:
            del self.active_conversations[from_addr]

        # Check if harmonization is needed
        if self.system.needs_harmonization():
            summary += f"\n\n⚠️ {self.system.labels_created_count - self.system.last_harmonization_count} labels pending harmonization"

        return summary

    def _format_quick_log_response(self, result):
        """Format a quick log response with label insights."""
        parts = []

        if result.get('extracted'):
            parts.append(f"✅ Logged {len(result['extracted'])} items")
        elif result.get('logged'):
            parts.append(f"✅ Logged to: {', '.join(result['logged'])}")

        if result.get('labels_generated'):
            parts.append(f"📏 {result['labels_generated']} labels created")

        if result.get('label_insights'):
            insights = result['label_insights']
            unique_count = sum(v.get('unique_concepts', 0) for v in insights.values())
            if unique_count > 0:
                parts.append(f"🎯 {unique_count} unique concepts detected")

        if parts:
            return '\n'.join(parts)
        else:
            return "Got your message! Reply 'help' for commands or 'chat' to start a conversation."

    def _get_help_text(self):
        """Get enhanced help text."""
        return """📚 BiggerBrother Commands:

CONVERSATION MODES:
• chat - Start an open-ended conversation
• checkin - Start a structured check-in
• detailed - Detailed activity logging mode

QUICK ACTIONS:
• status/summary - Get daily summary with insights
• pomodoro [task] - Start 25-min focus session
• skip/later - Skip current check-in
• done/exit - End current conversation
• help - Show this message

Just reply naturally in any mode!
I'll extract and log important information automatically.

Label Organization:
- Major themes are identified automatically
- Unique concepts are preserved
- Everything is intelligently grouped

System Features:
- Dual scheduler support for patterns & context
- EnhancedLabelHarmonizer for intelligent labeling
- Dynamic LogBook categories"""

    def _get_status_summary(self):
        """Get enhanced status summary with harmonization insights."""
        summary = self.system.get_daily_summary_with_logs()

        text = f"📊 Daily Summary - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n\n"

        # Add harmonization insights
        if hasattr(self.system, 'harmonizer'):
            insights = self.system.get_harmonization_insights()
            text += f"📏 Label Organization:\n"

            for category, stats in insights['categories'].items():
                text += f"  {category}: {stats['canonical_count']} groups "
                text += f"({stats['unique_concepts']} unique, {stats.get('top_groups_count', 0)} major)\n"

                # Show top groups if available
                if stats.get('top_groups_detail'):
                    for group_name, group_info in list(stats['top_groups_detail'].items())[:2]:
                        text += f"    • {group_name}: {group_info.get('frequency', 0):.0f} occurrences\n"

            if insights['needs_harmonization']:
                text += f"  ⚠️ Harmonization needed\n"
            text += "\n"

        # Active conversations
        if self.active_conversations:
            text += f"💬 Active Conversations: {len(self.active_conversations)}\n"
            for addr, conv in self.active_conversations.items():
                mode = conv['mode'].value
                duration = datetime.now(timezone.utc) - conv['start_time']
                text += f"  • {mode}: {int(duration.total_seconds() / 60)} min\n"
            text += "\n"

        text += f"Total logs: {summary['total_logs_today']}\n"
        text += f"Categories active: {summary['log_categories_active']}\n\n"

        if summary['log_books']:
            text += "Today's Activity:\n"
            for category, info in summary['log_books'].items():
                if info['count'] > 0:
                    text += f"  • {category}: {info['count']} entries\n"

        if summary.get('activity_metrics'):
            text += f"\nProductivity Score: {summary['activity_metrics']['productivity_score']:.2f}\n"
            text += f"Wellness Score: {summary['activity_metrics']['wellness_score']:.2f}\n"

        if summary.get('rpg'):
            text += f"\nCharacter Level: {summary['rpg']['level']}\n"
            text += f"XP Today: {summary['rpg']['xp_today']}\n"

        return text

    def _print_status(self):
        """Print enhanced system status."""
        print("\n" + "=" * 60)
        print("🧠 BiggerBrother System Active")
        print("   with EnhancedLabelHarmonizer & Dual Scheduler Support")
        print("=" * 60)
        print(f"📧 Gmail: {self.gmail.email_address}")
        print(f"📬 Monitoring: adaptive 15–300s (exponential backoff)")
        print(f"🔍 Filter: subject:BiggerBrother")

        categories = self.system.logbook.get_categories_for_context()
        print(f"\n📚 Log Categories: {len(categories)}")
        for cat in categories[:5]:
            print(f"   • {cat['name']}: {cat['entry_count']} entries")

        print("\n💬 Conversation Modes:")
        print("   • chat - Open-ended conversation")
        print("   • checkin - Structured check-in")
        print("   • detailed - Activity logging")

        print("\n🎯 Scheduler Types:")
        print("   • Adaptive - Pattern analysis & scheduling")
        print("   • Context-Aware - Message processing")

        # Show harmonization status
        if hasattr(self.system, 'harmonizer'):
            insights = self.system.get_harmonization_insights()
            print(f"\n📏 Label Groups:")
            for cat, stats in insights['categories'].items():
                print(f"   {cat}: {stats['canonical_count']} groups ({stats['unique_concepts']} unique)")

        if self.system.needs_harmonization():
            unharmonized = self.system.labels_created_count - self.system.last_harmonization_count
            print(f"\n⚠️ {unharmonized} labels need harmonization")

        print("\n📝 Send emails with 'BiggerBrother' in subject")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")

    def run_interactive(self):
        """Run with interactive CLI including chat mode."""
        self.start()

        try:
            while self.running:
                print("\n[1=quick, 2=check-in, 3=chat, 4=status, 5=test, 6=harmonize, 7=patterns, 8=schedule, h=help, q=quit]")

                choice = input("> ").strip().lower()

                if not choice:
                    continue

                elif choice == '1':
                    entry = input("Log entry: ")
                    result = self.system.process_message_with_context(entry)
                    print(self._format_quick_log_response(result))

                elif choice == '2':
                    # Start check-in (uses context scheduler internally)
                    session = self.system.start_check_in_with_logs()
                    print(f"\nAssistant: {session['greeting']}")
                    print("(Type 'done' to finish)\n")

                    while self.system.active_check_in:
                        user_input = input("You: ")

                        if user_input.lower() in ['done', 'exit']:
                            self.system.active_check_in = None
                            print("\nCheck-in complete!")
                            break

                        result = self.system.process_message_with_context(user_input)
                        print(f"\nAssistant: {result.get('response', 'I understand.')}")

                        # Show label insights
                        if result.get('label_insights'):
                            insights = result['label_insights']
                            unique_total = sum(v.get('unique_concepts', 0) for v in insights.values())
                            if unique_total > 0:
                                print(f"  [{unique_total} unique concepts detected]")

                elif choice == '3':
                    # Start chat mode
                    print("\n💬 Chat Mode - Let's have a conversation!")
                    print("(Type 'done' to finish)\n")

                    chat_messages = []  # keeps short context

                    while True:
                        user_input = input("You: ")
                        result = ""

                        if user_input.lower() in ['done', 'exit']:
                            print("\nChat ended. Thanks for the conversation!")
                            break

                        try:
                            result = self.system.process_message_with_context(user_input)
                            # OPTIONAL tiny footer to show logging happened, without leaking JSON
                            if result.get('labels_generated', 0) > 0:
                                print(f"  [{result['labels_generated']} topics noted]", end="")
                                print()
                        except Exception as e:
                            # keep chat resilient even if extractor hiccups
                            print(f"  [extractor error: {e}]")

                        # 1) Always generate a conversational reply with 4o
                        #    (use recent context; DO NOT depend on the extractor for text)
                            # 2) THEN: Generate conversational reply with context
                        try:
                            # keep last few turns for continuity
                            recent = chat_messages[-40:]

                            # Build context from similar messages
                            system_content = "Keep replies natural, and context-aware."
                            if result.get('similar_messages'):
                                # Take top N similar messages (limit total size)
                                similar_msgs = result['similar_messages'][:50]  # Top 5
                                if similar_msgs:
                                    system_content += "\n\nRelevant context from past conversations:\n"
                                    for i, msg in enumerate(similar_msgs, 1):
                                        # Truncate each message to avoid overwhelming
                                        truncated = msg[:10000] + "..." if len(msg) > 10000 else msg
                                        system_content += f"\n{i}. {truncated}"

                            response_text = self.system.openai_client.chat(
                                messages=[
                                    {"role": "system", "content": system_content},
                                    *recent,
                                    {"role": "user", "content": user_input}
                                ],
                                model="gpt-4o"
                            )
                        except Exception:
                            # If 4o fails, fall back to a safe human sentence
                            response_text = "I’m here. Keep going—what feels most alive in that thought?"

                        # 2) Print ONLY the human reply
                        #    Guard: if somehow we got JSON back, don’t dump it on the user
                        from json import loads as _loads
                        if isinstance(response_text, (dict, list)):
                            response_text = "Noted. Tell me more."
                        else:
                            s = (response_text or "").strip()
                            if s.startswith("{") or s.startswith("["):
                                try:
                                    _loads(s)  # it's JSON-like
                                    response_text = "Noted. Tell me more."
                                except Exception:
                                    pass

                        print(f"\nAssistant: {response_text}")

                        # 3) Update local chat context AFTER printing
                        chat_messages.append({"role": "user", "content": user_input})
                        chat_messages.append({"role": "assistant", "content": response_text})

                        # 4) Run your analyzer/extractor in the background of the UX
                        #    (it returns structured JSON; do NOT print it)


                elif choice == '4':
                    print(self._get_status_summary())

                elif choice == '5':
                    self.gmail.send_email(
                        to=self.gmail.email_address,
                        subject="BiggerBrother Test",
                        body="Test email sent! Reply with 'help' for commands or 'chat' to start a conversation."
                    )
                    print("✅ Test email sent!")

                elif choice == '6':
                    # Show harmonization status
                    insights = self.system.get_harmonization_insights()
                    print("\n📏 Harmonization Status:")
                    for category, stats in insights['categories'].items():
                        print(f"\n{category.upper()}:")
                        print(f"  Total groups: {stats['canonical_count']}")
                        print(f"  Unique concepts: {stats['unique_concepts']}")
                        print(f"  Major themes: {stats.get('top_groups_count', 0)}")

                        if stats.get('top_groups_detail'):
                            print("  Top groups:")
                            for group_name, group_info in list(stats['top_groups_detail'].items())[:3]:
                                print(f"    - {group_name}: {group_info.get('frequency', 0):.0f} occurrences")

                elif choice == '7':
                    # Analyze patterns using adaptive scheduler
                    print("Analyzing behavioral patterns...")
                    patterns = self.system.analyze_behavioral_patterns(days=30)
                    print(f"\n📊 Pattern Analysis:")
                    print(f"   Activity patterns found: {len(patterns.get('activity_patterns', []))}")
                    print(f"   Productivity cycles: {patterns.get('productivity_cycles', 'Unknown')}")
                    print(f"   Most active time: {patterns.get('most_active_time', 'Unknown')}")

                    # Show today's schedule
                    schedule = self.system.generate_daily_schedule()
                    print(f"\n📅 Today's Schedule:")
                    for item in schedule[:5]:
                        print(f"   {item.scheduled_time.strftime('%H:%M')} - {item.check_in_type.value}")

                elif choice == '8':
                    # Test scheduling
                    message = input("What do you need to schedule? ")
                    result = self.scheduling_system.process_message(message)

                    if result.get('scheduling', {}).get('intent_detected'):
                        print(f"\n{result['scheduling']['response']}")

                        # Get confirmation
                        confirm = input("\nYour choice: ")
                        session_id = result['scheduling']['session_id']
                        confirmation = self.scheduling_system.confirm_schedule(session_id, confirm)
                        print(f"\n{confirmation.get('message', 'Updated')}")
                    else:
                        print("No scheduling intent detected")

                elif choice == 'h':
                    print(self._get_help_text())

                elif choice == 'q':
                    break

        except KeyboardInterrupt:
            pass
        except Exception as e:
            self.logger.error(f"Interactive session error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        """Stop all services."""
        self.running = False
        self.orchestrator.stop()

        # Clear all active conversations
        for from_addr in list(self.active_conversations.keys()):
            self._end_conversation(from_addr)

        print("\n👋 BiggerBrother stopped")


def main():
    """Main entry point."""
    print("""
🧠 BiggerBrother with Gmail OAuth - Enhanced Edition
=====================================================
Now with EnhancedLabelHarmonizer, Dual Scheduler Support,
and 'just chatting' mode!
""")

    # Check for required files
    if not os.path.exists('credentials.json'):
        print("❌ credentials.json not found!")
        print("\nPlease follow Gmail OAuth setup first:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Enable Gmail API")
        print("3. Create OAuth credentials")
        print("4. Download as credentials.json")
        return

    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set!")
        print("\nSet your OpenAI API key:")
        print("export OPENAI_API_KEY='sk-...'")
        return

    # Create and run system
    system = BiggerBrotherEmailSystem()

    # Run interactive mode
    system.run_interactive()


if __name__ == "__main__":
    main()