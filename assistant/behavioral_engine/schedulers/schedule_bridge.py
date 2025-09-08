from __future__ import annotations
import os, json, uuid, smtplib
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import List, Dict, Optional, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class EmailNotifier:
    """Tiny SMTP wrapper. Enable by setting SMTP_* env vars + SMTP_TO/SMTP_FROM."""
    def __init__(self, env: dict, default_tz: str = "America/New_York"):
        self.tz = ZoneInfo(env.get("BB_TZ", default_tz))
        self.smtp_host = env.get("SMTP_HOST")
        self.smtp_port = int(env.get("SMTP_PORT", "587"))
        self.smtp_user = env.get("SMTP_USER")
        self.smtp_pass = env.get("SMTP_PASS")
        self.from_addr = env.get("SMTP_FROM")
        self.to_addr = env.get("SMTP_TO")
        self.enabled = all([self.smtp_host, self.smtp_user, self.smtp_pass, self.from_addr, self.to_addr])

    def send(self, subject: str, body: str) -> bool:
        if not self.enabled:
            return False
        msg = MIMEMultipart()
        msg["From"] = self.from_addr
        msg["To"] = self.to_addr
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as s:
                s.starttls()
                s.login(self.smtp_user, self.smtp_pass)
                s.sendmail(self.from_addr, [self.to_addr], msg.as_string())
            return True
        except Exception as e:
            print(f"[warn] email send failed: {e}")
            return False


class ReminderDaemon:
    """
    Simple 'tick' loop: read reminders file and send any that are due.
    Call .tick() inside your main request flow—no background thread required.
    """
    def __init__(self, reminders_file: Path, notifier: EmailNotifier):
        self.reminders_file = reminders_file
        self.notifier = notifier

    def tick(self):
        if not self.reminders_file.exists():
            return
        try:
            lines = self.reminders_file.read_text(encoding="utf-8").splitlines()
            items = [json.loads(l) for l in lines if l.strip()]
        except Exception:
            return

        now_utc = datetime.now(timezone.utc)
        out = []
        changed = False
        for it in items:
            send_at = it.get("send_at_utc")
            if isinstance(send_at, str):
                try:
                    send_dt = datetime.fromisoformat(send_at.replace("Z", "+00:00"))
                except Exception:
                    send_dt = None
            else:
                send_dt = None

            if not it.get("sent") and send_dt and now_utc >= send_dt:
                ok = self.notifier.send(it.get("subject", "Reminder"), it.get("body", ""))
                it["sent"] = bool(ok)
                it["sent_at_utc"] = datetime.now(timezone.utc).isoformat()
                changed = True
            out.append(json.dumps(it, ensure_ascii=False))

        if changed:
            self.reminders_file.write_text("\n".join(out), encoding="utf-8")


class ScheduleBridge:
    """
    Collects schedule-worthy notes during chat, suggests tasks for a day,
    and crystallizes a day plan with email reminders at task start.
    """
    def __init__(self, planner_dir: str, openai_client=None, mail_env: dict | None = None):
        self.planner_dir = Path(planner_dir)
        self.daily_dir = self.planner_dir / "daily"
        self.outbox_dir = self.planner_dir / "outbox"
        self.notes_file = self.planner_dir / "notes.jsonl"
        self.reminders_file = self.planner_dir / "reminders.jsonl"
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self.openai_client = openai_client
        self.tz = ZoneInfo(os.getenv("BB_TZ", "America/New_York"))
        self.notifier = EmailNotifier(mail_env or os.environ)
        self.reminders = ReminderDaemon(self.reminders_file, self.notifier)

    def capture_note(self, text: str, harmonized: Optional[Dict] = None, source: str = "chat") -> Dict:
        note = {
            "id": f"note_{uuid.uuid4().hex[:8]}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "text": text,
            "labels": harmonized or {},
            "parsed": None,
            "scheduled": False
        }
        # Optional task parsing via your OpenAI wrapper
        if self.openai_client is not None:
            try:
                prompt = ("Extract candidate tasks from the text. "
                          "Return JSON array of {title, earliest_date (YYYY-MM-DD|null), "
                          "preferred_time (HH:MM|null), duration_min (int, default 30)}.")
                raw = self.openai_client.chat(
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text}
                    ],
                    model="gpt-5-nano"
                ).strip()
                if "```json" in raw:
                    raw = raw.split("```json")[1].split("```")[0]
                parsed = json.loads(raw) if raw.startswith("[") else [json.loads(raw)]
                norm = []
                for t in parsed or []:
                    norm.append({
                        "title": t.get("title") or text[:80],
                        "earliest_date": t.get("earliest_date"),
                        "preferred_time": t.get("preferred_time"),
                        "duration_min": int(t.get("duration_min") or 30)
                    })
                note["parsed"] = norm
            except Exception as e:
                print(f"[warn] schedule note parse failed: {e}")

        with self.notes_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(note, ensure_ascii=False) + "\n")
        return note

    def get_suggestions_for_day(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        date = date or datetime.now(self.tz)
        day_str = date.date().isoformat()
        notes = []
        if self.notes_file.exists():
            for line in self.notes_file.read_text(encoding="utf-8").splitlines():
                try:
                    obj = json.loads(line)
                    if not obj.get("scheduled"):
                        notes.append(obj)
                except Exception:
                    pass

        tasks = []
        for n in notes:
            use = n.get("parsed") or [{"title": n.get("text"), "duration_min": 30}]
            for t in use:
                earliest = t.get("earliest_date") or day_str
                if earliest <= day_str:
                    tasks.append({
                        "note_id": n["id"],
                        "title": t.get("title") or n["text"][:80],
                        "duration_min": int(t.get("duration_min") or 30),
                        "preferred_time": t.get("preferred_time"),
                    })
        return {"date": day_str, "suggested": tasks}

    def crystallize_schedule(self, tasks: List[Dict], date: Optional[datetime] = None,
                             tz: Optional[str] = None, send_emails: bool = True) -> Dict[str, Any]:
        tz = ZoneInfo(tz) if tz else self.tz
        date = date or datetime.now(tz)
        day_str = date.date().isoformat()

        # Pack tasks into a sequential day, respecting preferred_time when given
        cur_time = datetime.combine(date.date(), datetime.min.time()).replace(tzinfo=tz, hour=9, minute=0)
        events = []
        for t in tasks:
            start = cur_time
            if t.get("preferred_time"):
                try:
                    hh, mm = map(int, t["preferred_time"].split(":"))
                    start = start.replace(hour=hh, minute=mm)
                except Exception:
                    pass
            duration = int(t.get("duration_min") or 30)
            end = start + timedelta(minutes=duration)
            events.append({
                "title": t.get("title", "Task"),
                "start_local": start.isoformat(),
                "end_local": end.isoformat(),
                "duration_min": duration
            })
            cur_time = end + timedelta(minutes=5)

        # Persist day file
        day_file = self.daily_dir / f"{day_str}.json"
        day_file.write_text(json.dumps({"date": day_str, "tz": str(tz), "events": events}, indent=2), encoding="utf-8")

        # Mark notes as scheduled
        if self.notes_file.exists():
            note_ids = {t.get("note_id") for t in tasks if t.get("note_id")}
            lines = self.notes_file.read_text(encoding="utf-8").splitlines()
            out = []
            for line in lines:
                try:
                    obj = json.loads(line)
                    if obj.get("id") in note_ids:
                        obj["scheduled"] = True
                        obj["scheduled_for"] = day_str
                    out.append(json.dumps(obj, ensure_ascii=False))
                except Exception:
                    out.append(line)
            self.notes_file.write_text("\n".join(out), encoding="utf-8")

        # Queue email reminders (one at the start of each task)
        if send_emails:
            to_append = []
            for e in events:
                start_local = datetime.fromisoformat(e["start_local"])
                start_utc = start_local.astimezone(timezone.utc)
                subj = f"[Schedule] {e['title']} — starts now"
                body = (
                    f"Task: {e['title']}\n"
                    f"Start (local {tz}): {start_local.strftime('%Y-%m-%d %H:%M')}\n"
                    f"Duration: {e['duration_min']} min\n"
                )
                to_append.append(json.dumps({
                    "subject": subj,
                    "body": body,
                    "send_at_utc": start_utc.isoformat(),
                    "sent": False
                }, ensure_ascii=False))
            with self.reminders_file.open("a", encoding="utf-8") as f:
                if to_append:
                    f.write("\n".join(to_append) + "\n")

            # One sweep now—for any overdue reminders
            self.reminders.tick()

        return {"date": day_str, "events": events, "reminders_file": str(self.reminders_file)}
