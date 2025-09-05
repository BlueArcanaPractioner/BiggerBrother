"""
Intelligent Scheduling System for BiggerBrother
Integrates intent parsing, schedule planning, and confirmation workflows
"""

from __future__ import annotations
import json
import re
from datetime import datetime, timedelta, time, timezone
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field
import logging

# Import your existing modules
from app.openai_client import OpenAIClient
from assistant.behavioral_engine.schedulers.adaptive_scheduler import AdaptiveScheduler
from assistant.behavioral_engine.logbooks.dynamic_logbook_system import DynamicLogBook


class ScheduleIntentType(Enum):
    """Types of scheduling intents we can detect"""
    APPOINTMENT = "appointment"
    TODO = "todo"
    REMINDER = "reminder"
    ROUTINE = "routine"
    PLAN_DAY = "plan_day"
    MODIFY_SCHEDULE = "modify_schedule"
    CHECK_AVAILABILITY = "check_availability"
    NONE = "none"


@dataclass
class BusinessHours:
    """Represents business hours for a location/service"""
    name: str
    monday: Tuple[time, time] = None
    tuesday: Tuple[time, time] = None
    wednesday: Tuple[time, time] = None
    thursday: Tuple[time, time] = None
    friday: Tuple[time, time] = None
    saturday: Tuple[time, time] = None
    sunday: Tuple[time, time] = None
    
    def is_open(self, dt: datetime) -> bool:
        """Check if business is open at given datetime"""
        weekday = dt.weekday()
        hours_map = {
            0: self.monday, 1: self.tuesday, 2: self.wednesday,
            3: self.thursday, 4: self.friday, 5: self.saturday, 6: self.sunday
        }
        
        hours = hours_map.get(weekday)
        if not hours:
            return False
            
        current_time = dt.time()
        return hours[0] <= current_time <= hours[1]
    
    def next_available_slot(self, preferred_dt: datetime, duration_minutes: int = 60) -> datetime:
        """Find next available time slot after preferred datetime"""
        dt = preferred_dt
        max_days_ahead = 14
        
        for _ in range(max_days_ahead * 24):  # Check hourly for 2 weeks
            if self.is_open(dt):
                # Check if entire duration fits in business hours
                end_time = dt + timedelta(minutes=duration_minutes)
                if self.is_open(end_time):
                    return dt
            
            # Try next hour
            dt = dt + timedelta(hours=1)
            
            # If we've moved to next day, jump to opening time
            if dt.date() != preferred_dt.date():
                weekday = dt.weekday()
                hours_map = {
                    0: self.monday, 1: self.tuesday, 2: self.wednesday,
                    3: self.thursday, 4: self.friday, 5: self.saturday, 6: self.sunday
                }
                hours = hours_map.get(weekday)
                if hours:
                    dt = datetime.combine(dt.date(), hours[0])
        
        return None  # No slot found in next 2 weeks


@dataclass
class ScheduleIntent:
    """Parsed scheduling intent from conversation"""
    intent_type: ScheduleIntentType
    subject: str  # What needs to be scheduled
    preferred_time: Optional[datetime] = None
    constraints: List[str] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    location: Optional[str] = None
    duration_minutes: Optional[int] = None
    priority: int = 1  # 1-5, 5 being highest
    notes: str = ""
    business_hours_required: Optional[str] = None  # e.g., "chiropractor"
    
    def to_dict(self) -> Dict:
        return {
            'intent_type': self.intent_type.value,
            'subject': self.subject,
            'preferred_time': self.preferred_time.isoformat() if self.preferred_time else None,
            'constraints': self.constraints,
            'participants': self.participants,
            'location': self.location,
            'duration_minutes': self.duration_minutes,
            'priority': self.priority,
            'notes': self.notes,
            'business_hours_required': self.business_hours_required
        }


class IntentParser:
    """Parses scheduling intents from natural language using GPT-5-nano"""
    
    def __init__(self, openai_client: OpenAIClient):
        self.client = openai_client
        self.logger = logging.getLogger(__name__)
        
        # Common business hours (you can load these from a config file)
        self.business_hours_db = {
            'chiropractor': BusinessHours(
                name='chiropractor',
                monday=(time(9, 0), time(18, 0)),
                tuesday=(time(9, 0), time(18, 0)),
                wednesday=(time(9, 0), time(18, 0)),
                thursday=(time(9, 0), time(18, 0)),
                friday=(time(9, 0), time(17, 0)),
                saturday=None,  # Closed
                sunday=None     # Closed
            ),
            'doctor': BusinessHours(
                name='doctor',
                monday=(time(8, 0), time(17, 0)),
                tuesday=(time(8, 0), time(17, 0)),
                wednesday=(time(8, 0), time(17, 0)),
                thursday=(time(8, 0), time(17, 0)),
                friday=(time(8, 0), time(16, 0)),
                saturday=(time(9, 0), time(12, 0)),
                sunday=None
            )
        }
    
    def parse_intent(self, message: str, context: Optional[Dict] = None) -> ScheduleIntent:
        """Parse scheduling intent from message using GPT-5-nano (JSON mode)."""
        system = ("Extract a single scheduling intent as JSON. Keys: "
                  "intent_type (appointment|todo|reminder|routine|plan_day|modify_schedule|check_availability|none), "
                  "subject, preferred_time (ISO or null), constraints (array), participants (array), "
                  "location, duration_minutes (int), priority (1-5), notes, business_hours_required (string or null). "
                  "If text bans late times (e.g., 'not at 7pm', 'after 7pm closed'), include constraint 'NOT_AFTER=19:00'. "
                  "If weekends are excluded, include 'WEEKDAY_ONLY'.")
        user = f"Message: {message}\nContext: {json.dumps(context) if context else 'null'}"
        try:
            resp = self.client.chat(
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                model="gpt-5-nano",
                response_format={"type":"json_object"}
            )
            data = resp if isinstance(resp, dict) else self._extract_json(resp)
            data = self._postprocess_constraints(message, data)
            return self._create_intent(data)
            
        except Exception as e:
            self.logger.error(f"Failed to parse intent: {e}")
            return ScheduleIntent(
                intent_type=ScheduleIntentType.NONE,
                subject=message[:50]  # Use first 50 chars as fallback
            )

    def _postprocess_constraints(self, message: str, data: Dict) -> Dict:
        constraints = [c.lower() for c in (data.get('constraints') or [])]
        txt = (message or "").lower()
        if "weekend" in txt or "weekends" in txt:
            constraints.append("WEEKEND_MENTIONED")
            if "closed" in txt or "not on weekend" in txt or "not on weekends" in txt:
                constraints.append("WEEKDAY_ONLY")
        m = re.search(r'(\b\d{1,2})(?::(\d{2}))?\s*(am|pm)\b', txt)
        if m and ("not" in txt or "after" in txt):
            hh = int(m.group(1)) % 12
            mm = int(m.group(2) or "0")
            if m.group(3) == "pm": hh += 12
            constraints.append(f"NOT_AFTER={hh:02d}:{mm:02d}")
        data['constraints'] = list(dict.fromkeys(constraints))
        if not data.get('business_hours_required'):
            for prov in ("chiropractor","dentist","doctor","clinic","bank","post office"):
                if prov in txt:
                    data['business_hours_required'] = prov
                    break
        return data

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON object from text response"""
        # Try to find JSON in the response
        json_match = re.search(r'\{.*}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to parse entire response
        try:
            return json.loads(text)
        except:
            return {}
    
    def _create_intent(self, data: Dict) -> ScheduleIntent:
        """Create ScheduleIntent from parsed data"""
        intent_type = ScheduleIntentType.NONE
        try:
            intent_type = ScheduleIntentType(data.get('intent_type', 'none'))
        except ValueError:
            pass
        
        # Parse preferred time if present
        preferred_time = None
        if data.get('preferred_time'):
            try:
                preferred_time = datetime.fromisoformat(data['preferred_time'])
            except:
                pass
        
        return ScheduleIntent(
            intent_type=intent_type,
            subject=data.get('subject', ''),
            preferred_time=preferred_time,
            constraints=data.get('constraints', []),
            participants=data.get('participants', []),
            location=data.get('location'),
            duration_minutes=data.get('duration_minutes'),
            priority=data.get('priority', 3),
            notes=data.get('notes', ''),
            business_hours_required=data.get('business_hours_required')
        )


class ScheduleProposer:
    """Proposes schedules based on intents and context"""
    
    def __init__(self, openai_client: OpenAIClient, logbook: DynamicLogBook):
        self.client = openai_client
        self.logbook = logbook
        self.logger = logging.getLogger(__name__)
    
    def load_relevant_context(self, intent: ScheduleIntent) -> Dict:
        """Load relevant logs and context for scheduling decision"""
        context = {
            'recent_schedule': [],
            'similar_appointments': [],
            'daily_patterns': [],
            'constraints': []
        }
        
        # Search logbooks for relevant entries
        if intent.business_hours_required:
            # Look for past appointments with same service
            similar = self.logbook.search_logs(
                query=intent.business_hours_required,
                categories=['appointments', 'schedule'],
                date_range=(datetime.now(timezone.utc) - timedelta(days=90), datetime.now(timezone.utc))
            )
            context['similar_appointments'] = similar[:5]  # Last 5 similar
        
        # Get current week's schedule
        week_logs = self.logbook.search_logs(
            query='schedule appointment meeting',
            date_range=(datetime.now(timezone.utc) - timedelta(days=7), datetime.now(timezone.utc) + timedelta(days=7))
        )
        context['recent_schedule'] = week_logs
        
        return context
    
    def propose_schedule(self, intent: ScheduleIntent, context: Dict) -> Dict:
        """Generate schedule proposal based on intent and context"""
        
        # Load business hours if needed
        business_hours = None
        if intent.business_hours_required:
            from assistant.behavioral_engine.schedulers.adaptive_scheduler import AdaptiveScheduler
            business_hours = self._get_business_hours(intent.business_hours_required)
        
        # Build proposal prompt for conversationalist
        proposal_prompt = f"""Based on the following scheduling request and context, propose a schedule.

Request: {intent.subject}
Type: {intent.intent_type.value}
Constraints: {', '.join(intent.constraints) if intent.constraints else 'None'}
Priority: {intent.priority}/5
Duration: {intent.duration_minutes} minutes

Context from past schedules:
{json.dumps(context.get('similar_appointments', []), indent=2)}

Recent schedule:
{json.dumps(context.get('recent_schedule', []), indent=2)}

Business Hours for {intent.business_hours_required if intent.business_hours_required else 'N/A'}:
{self._format_business_hours(business_hours) if business_hours else 'No business hours constraints'}

Propose 3 time slots that would work well. Consider:
1. Business hours constraints
2. User's typical patterns
3. The stated constraints
4. Priority level

Format as:
Option 1: [Day, Date, Time] - [Reason why this works]
Option 2: [Day, Date, Time] - [Reason why this works]
Option 3: [Day, Date, Time] - [Reason why this works]

Then add a recommendation for which option is best and why."""

        proposed_times = self._compute_proposed_slots(intent, business_hours)
        if proposed_times:
            lines = []
            for i, p in enumerate(proposed_times, 1):
                reason = "within business hours"
                if "WEEKDAY_ONLY" in (intent.constraints or []):
                    reason += ", weekday"
                lines.append(f"Option {i}: {p['human']} - {reason}")
            recommendation = "I recommend Option 1 so you can address this sooner."
            proposal = "\n".join(lines + ["", recommendation])
        else:
            # Add a Monday TODO if business is closed
            next_mon = (datetime.now(timezone.utc).date() + timedelta(days=((7 - datetime.now(timezone.utc).weekday()) % 7 or 7)))
            self.logbook.log_entry(
                category_name='todos',
                data={'text': f"Call {intent.business_hours_required or 'provider'} to schedule",
                      'due_date': next_mon.isoformat(),
                      'priority': intent.priority},
                extracted_by='scheduling_system'
            )
            proposal = (f"I couldn't find any slots that meet the constraints in the next two weeks.\n"
                        f"I added a TODO for Monday ({next_mon.isoformat()}) to call the "
                        f"{intent.business_hours_required or 'provider'}.")
        
        return {
            'intent': intent.to_dict(),
            'proposal_text': proposal,
            'proposed_times': proposed_times,
            'context_used': {
                'similar_count': len(context.get('similar_appointments', [])),
                'schedule_items': len(context.get('recent_schedule', []))
            }
        }

    def _compute_proposed_slots(self, intent: ScheduleIntent, hours: Optional[BusinessHours]) -> List[Dict]:
        duration = intent.duration_minutes or (60 if intent.intent_type == ScheduleIntentType.APPOINTMENT else 30)
        not_after, weekday_only = None, False
        for c in (intent.constraints or []):
            if c.startswith("NOT_AFTER="):
                try:
                    hh, mm = c.split("=",1)[1].split(":")
                    not_after = time(int(hh), int(mm))
                except: pass
            if c == "WEEKDAY_ONLY":
                weekday_only = True
        now = datetime.now(timezone.utc)
        start = intent.preferred_time or (now + timedelta(hours=1))
        out, day, checked = [], start, 0
        while len(out) < 3 and checked < 14:
            if weekday_only and day.weekday() >= 5:
                day = (day + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
                checked += 1; continue
            if hours:
                m = {0:hours.monday,1:hours.tuesday,2:hours.wednesday,3:hours.thursday,4:hours.friday,5:hours.saturday,6:hours.sunday}[day.weekday()]
                if not m:
                    day = (day + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
                    checked += 1; continue
                open_t, close_t = m
            else:
                open_t, close_t = time(9,0), time(17,0)
            if not_after:
                close_t = min(close_t, not_after)
            earliest = max(datetime.combine(day.date(), open_t), day)
            latest_start = datetime.combine(day.date(), close_t) - timedelta(minutes=duration)
            if earliest <= latest_start:
                candidate = earliest
                out.append({"text": candidate.strftime("%A, %B %d at %I:%M %p"),
                            "human": candidate.strftime("%A, %B %d at %I:%M %p"),
                            "parsed": candidate.isoformat()})
            day = (day + timedelta(days=1)).replace(hour=open_t.hour, minute=open_t.minute, second=0, microsecond=0)
            checked += 1
        return out
    
    def _get_business_hours(self, business_type: str) -> Optional[BusinessHours]:
        """Get business hours for a service type"""
        # In a real implementation, this would query a database or API
        parser = IntentParser(self.client)
        return parser.business_hours_db.get(business_type.lower())
    
    def _format_business_hours(self, hours: BusinessHours) -> str:
        """Format business hours for display"""
        if not hours:
            return "No hours available"
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        hours_list = [hours.monday, hours.tuesday, hours.wednesday, hours.thursday, 
                     hours.friday, hours.saturday, hours.sunday]
        
        formatted = []
        for day, day_hours in zip(days, hours_list):
            if day_hours:
                formatted.append(f"{day}: {day_hours[0].strftime('%I:%M %p')} - {day_hours[1].strftime('%I:%M %p')}")
            else:
                formatted.append(f"{day}: Closed")
        
        return '\n'.join(formatted)
    
    def _parse_proposed_times(self, proposal_text: str) -> List[Dict]:
        """Parse proposed times from conversationalist response"""
        # Simple regex to find patterns like "Monday, January 15, 2:00 PM"
        pattern = r'Option \d+: ([^-\n]+)'
        matches = re.findall(pattern, proposal_text)
        
        proposed = []
        for match in matches[:3]:  # Get up to 3 options
            proposed.append({
                'text': match.strip(),
                'parsed': self._parse_datetime_string(match.strip())
            })
        
        return proposed
    
    def _parse_datetime_string(self, dt_string: str) -> Optional[str]:
        """Try to parse a datetime string into ISO format"""
        # This is simplified - in production you'd use a more robust parser
        # For now, return None and let GPT-5-nano handle the parsing
        return None


class IntelligentSchedulingSystem:
    """Main orchestrator for intelligent scheduling within BiggerBrother"""
    
    def __init__(self, openai_client=None, complete_system=None, base_dir: str = None):
        # Use provided instances or create new ones
        self.openai_client = openai_client or OpenAIClient()
        self.complete_system = complete_system  # Will be set by gmail_runner
        self.email_system = None  # Will be set by gmail_runner
        
        # Initialize scheduling components
        self.intent_parser = IntentParser(self.openai_client)
        self.schedule_proposer = None  # Will be initialized when complete_system is set
        
        # Track conversation state
        self.active_scheduling_session = None
        self.pending_confirmations = {}
        
        self.logger = logging.getLogger(__name__)

    def set_complete_system(self, complete_system):
        """Set the complete system reference to avoid circular imports"""
        self.complete_system = complete_system
        # Now we can initialize the schedule proposer
        self.schedule_proposer = ScheduleProposer(
            self.openai_client,
            self.complete_system.logbook
        )

    
    def process_message(self, message: str, source: str = 'email') -> Dict:
        """Process incoming message for scheduling intents"""
        
        # First, let complete system process for labels and context
        base_result = self.complete_system.process_message_with_context(message)
        
        # Parse for scheduling intent
        intent = self.intent_parser.parse_intent(
            message,
            context=base_result.get('harmonized_labels')
        )
        
        # If no scheduling intent, return base result
        if intent.intent_type == ScheduleIntentType.NONE:
            return {
                **base_result,
                'scheduling': {'intent_detected': False}
            }
        
        self.logger.info(f"Detected scheduling intent: {intent.intent_type.value}")
        
        # Load relevant context from logbooks
        context = self.schedule_proposer.load_relevant_context(intent)
        
        # Generate schedule proposal
        proposal = self.schedule_proposer.propose_schedule(intent, context)
        
        # Store for confirmation
        session_id = datetime.now().isoformat()
        self.pending_confirmations[session_id] = proposal
        
        # Generate response for user
        response = self._generate_user_response(proposal, session_id)
        
        return {
            **base_result,
            'scheduling': {
                'intent_detected': True,
                'intent': intent.to_dict(),
                'proposal': proposal,
                'session_id': session_id,
                'response': response
            }
        }
    
    def confirm_schedule(self, session_id: str, user_response: str) -> Dict:
        """Process user confirmation or adjustment of proposed schedule"""
        
        if session_id not in self.pending_confirmations:
            return {'error': 'Session not found'}
        
        proposal = self.pending_confirmations[session_id]
        
        # Parse user response for confirmation or adjustments
        confirm_prompt = f"""Parse the user's response to a schedule proposal.

Original Proposal:
{proposal['proposal_text']}

User Response: "{user_response}"

Determine:
1. confirmed: true if user agrees, false if they want changes
2. selected_option: which option they chose (1, 2, or 3), or null
3. adjustments: any requested changes
4. final_datetime: the final agreed datetime in ISO format

Return JSON only."""

        response = self.openai_client.complete(
            prompt=confirm_prompt,
            model='gpt-5-mini',
            max_tokens=300
        )
        
        confirmation = self.intent_parser._extract_json(response)
        
        if confirmation.get('confirmed'):
            # Add to schedule
            result = self._add_to_schedule(proposal, confirmation)
            
            # Clean up session
            del self.pending_confirmations[session_id]
            
            # Log to logbook
            self.complete_system.logbook.log_entry(
                category_name='appointments',
                data={
                    'type': proposal['intent']['intent_type'],
                    'subject': proposal['intent']['subject'],
                    'datetime': confirmation.get('final_datetime'),
                    'duration': proposal['intent']['duration_minutes'],
                    'location': proposal['intent']['location'],
                    'notes': proposal['intent']['notes']
                },
                raw_text=user_response,
                extracted_by='scheduling_system',
                confidence=0.95
            )
            
            return {
                'status': 'confirmed',
                'scheduled': result,
                'message': f"✓ Scheduled: {proposal['intent']['subject']}"
            }
        else:
            # Need to repropose with adjustments
            return self._handle_adjustments(session_id, confirmation)
    
    def _generate_user_response(self, proposal: Dict, session_id: str) -> str:
        """Generate conversational response with schedule proposal"""
        
        response = f"""I understand you need to schedule {proposal['intent']['subject']}.
        
{proposal['proposal_text']}

To confirm one of these options, just let me know which works best for you, 
or suggest a different time if none of these work.

(Session: {session_id[:8]}...)"""
        
        return response
    
    def _add_to_schedule(self, proposal: Dict, confirmation: Dict) -> Dict:
        """Add confirmed appointment to schedule"""
        
        # Parse final datetime
        final_dt = None
        if confirmation.get('final_datetime'):
            try:
                final_dt = datetime.fromisoformat(confirmation['final_datetime'])
            except:
                final_dt = datetime.now(timezone.utc) + timedelta(days=1)  # Default to tomorrow
        
        # Create schedule entry
        schedule_entry = {
            'datetime': final_dt,
            'subject': proposal['intent']['subject'],
            'duration_minutes': proposal['intent']['duration_minutes'],
            'type': proposal['intent']['intent_type'],
            'priority': proposal['intent']['priority'],
            'location': proposal['intent']['location'],
            'participants': proposal['intent']['participants'],
            'notes': proposal['intent']['notes']
        }
        
        # Add to adaptive scheduler
        if final_dt:
            if self.complete_system and hasattr(self.complete_system, 'adaptive_scheduler'):
                self.complete_system.adaptive_scheduler.generate_daily_schedule(final_dt)
            if self.email_system:
                subj = f"Appointment: {proposal['intent']['subject']}"
                body = f"{proposal['intent']['subject']} at {final_dt.strftime('%Y-%m-%d %H:%M')}"
                from threading import Timer
                # schedule 1h before and at start (assume local wall time)
                try:
                    delay1 = max(0, (final_dt - timedelta(hours=1) - datetime.now(timezone.utc)).total_seconds())
                    Timer(delay1, lambda: self.email_system.notifier._send_email("Reminder (1h)", body)).start()
                    delay2 = max(0, (final_dt - datetime.now(timezone.utc)).total_seconds())
                    Timer(delay2, lambda: self.email_system.notifier._send_email("Start", body)).start()
                except Exception as e:
                    self.logger.error(f"Failed to schedule email reminders: {e}")
        
        return schedule_entry
    
    def _handle_adjustments(self, session_id: str, adjustments: Dict) -> Dict:
        """Handle user's requested adjustments to proposal"""
        
        # Regenerate proposal with adjustments
        proposal = self.pending_confirmations[session_id]
        
        # Update intent with adjustments
        if adjustments.get('adjustments'):
            proposal['intent']['constraints'].extend(adjustments['adjustments'])
        
        # Repropose
        new_proposal = self.schedule_proposer.propose_schedule(
            ScheduleIntent(**proposal['intent']),
            proposal.get('context_used', {})
        )
        
        # Update session
        self.pending_confirmations[session_id] = new_proposal
        
        return {
            'status': 'adjusted',
            'response': self._generate_user_response(new_proposal, session_id)
        }
    
    def run_email_loop(self):
        """Run the email processing loop with scheduling integration"""
        
        self.logger.info("Starting intelligent scheduling email loop...")
        
        while True:
            try:
                # Check for new emails
                messages = self.email_system.gmail.get_unread_messages()
                
                for msg in messages:
                    # Extract message content
                    msg_data = self.email_system.gmail._extract_message_data(msg)
                    body = msg_data.get('body', '')
                    
                    # Process for scheduling
                    result = self.process_message(body, source='email')
                    
                    # Generate response
                    if result['scheduling']['intent_detected']:
                        response_text = result['scheduling']['response']
                    else:
                        # Use regular conversational response
                        response_text = self.email_system.system.process_message(body)
                    
                    # Send response
                    self.email_system.gmail.send_email(
                        service=self.email_system.gmail.service,
                        to_addr=msg_data.get('from'),
                        subject=f"Re: {msg_data.get('subject')}",
                        text_body=response_text,
                        thread_id=msg_data.get('thread_id')
                    )
                    
                    # Mark as read
                    self.email_system.gmail.mark_as_read(
                        self.email_system.gmail.service,
                        msg['id']
                    )
                
                # Sleep before next check
                import time
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                self.logger.info("Stopping email loop...")
                break
            except Exception as e:
                self.logger.error(f"Error in email loop: {e}")
                import time
                time.sleep(60)  # Wait longer on error


# Example usage
if __name__ == "__main__":
    # Initialize the system
    scheduler = IntelligentSchedulingSystem(
        base_dir="data",
        use_sandbox=False
    )
    
    # Example: Process a scheduling request
    message = "My neck is starting to hurt. I'll need to schedule an appointment with my chiropractor. If it is at 7pm, or the weekend the chiropractor will be closed."
    
    result = scheduler.process_message(message)
    
    if result['scheduling']['intent_detected']:
        print(f"Detected intent: {result['scheduling']['intent']}")
        print(f"\nProposal:\n{result['scheduling']['response']}")
        
        # Simulate user confirmation
        user_response = "Option 1 works great for me"
        confirmation = scheduler.confirm_schedule(
            result['scheduling']['session_id'],
            user_response
        )
        print(f"\nConfirmation: {confirmation}")
    
    # Or run the email loop
    # scheduler.run_email_loop()
