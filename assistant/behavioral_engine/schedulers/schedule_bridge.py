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
    def __init__(self, env: dict, default_tz: str = "America/New_York"):
        self.tz = ZoneInfo(env.get("BB_TZ", default_tz))
        self.smtp_host = env.get("SMTP_HOST"); self.smtp_port = int(env.get("SMTP_PORT", "587"))
        self.smtp_user = env.get("SMTP_USER"); self.smtp_pass = env.get("SMTP_PASS")
        self.from_addr = env.get("SMTP_FROM"); self.to_addr = env.get("SMTP_TO")
        self.enabled = all([self.smtp_host, self.smtp_user, self.smtp_pass, self.from_addr, self.to_addr])

    def send(self, subject: str, body: str) -> bool:
        if not self.enabled: return False
        msg = MIMEMultipart()
        msg["From"] = self.from_addr; msg["To"] = self.to_addr; msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as s:
                s.starttls(); s.login(self.smtp_user, self.smtp_pass); s.sendmail(self.from_addr, [self.to_addr], msg.as_string())
            return True
        except Exception as e:
            print(f"[warn] email send failed: {e}")
            return False

class ReminderDaemon:
    def __init__(self, reminders_file: Path, notifier: EmailNotifier):
        self.reminders_file = reminders_file; self.notifier = notifier
    def tick(self):
        if not self.reminders_file.exists(): return
        try:
            items = [json.loads(l) for l in self.reminders_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        except Exception: return
        now_utc = datetime.now(timezone.utc); out = []; changed = False
        for it in items:
            send_dt = None
            s = it.get("send_at_utc");
            if isinstance(s, str):
                try: send_dt = datetime.fromisoformat(s.replace("Z","+00:00"))
                except Exception: send_dt = None
            if not it.get("sent") and send_dt and now_utc >= send_dt:
                ok = self.notifier.send(it.get("subject","Reminder"), it.get("body",""))
                it["sent"] = bool(ok); it["sent_at_utc"] = datetime.now(timezone.utc).isoformat(); changed = True
            out.append(json.dumps(it, ensure_ascii=False))
        if changed: self.reminders_file.write_text("\n".join(out), encoding="utf-8")

class ScheduleBridge:
    """
    Notes → draft schedule (during planning chat) → crystallized day plan + email reminders.
    """
    def __init__(self, planner_dir: str, openai_client=None, mail_env: dict | None = None):
        self.planner_dir = Path(planner_dir); self.daily_dir = self.planner_dir / "daily"
        self.outbox_dir = self.planner_dir / "outbox"; self.drafts_dir = self.planner_dir / "drafts"
        self.notes_file = self.planner_dir / "notes.jsonl"; self.reminders_file = self.planner_dir / "reminders.jsonl"
        for p in (self.daily_dir, self.outbox_dir, self.drafts_dir): p.mkdir(parents=True, exist_ok=True)
        self.openai_client = openai_client
        self.tz = ZoneInfo(os.getenv("BB_TZ", "America/New_York"))
        self.notifier = EmailNotifier(mail_env or os.environ); self.reminders = ReminderDaemon(self.reminders_file, self.notifier)

    # ---- Draft handling ----
    def _draft_path(self, day_str: str) -> Path: return self.drafts_dir / f"{day_str}.json"
    def get_draft(self, day: Optional[datetime] = None) -> Dict[str, Any]:
        d = (day or datetime.now(self.tz)).date().isoformat()
        p = self._draft_path(d)
        if p.exists():
            try: return json.loads(p.read_text(encoding="utf-8"))
            except Exception: pass
        return {"date": d, "tz": str(self.tz), "tasks": []}

    def append_suggestions(self, suggestions: List[Dict], day: Optional[datetime] = None) -> Dict[str, Any]:
        d = (day or datetime.now(self.tz)).date().isoformat()
        draft = self.get_draft(day)
        # normalize + merge
        seen = {(t["title"], t.get("preferred_time"), int(t.get("duration_min",30))) for t in draft["tasks"]}
        for s in suggestions or []:
            title = s.get("title") or "Task"; dur = int(s.get("duration_min") or 30)
            pt = s.get("preferred_time")
            key = (title, pt, dur)
            if key in seen: continue
            draft["tasks"].append({"title": title, "duration_min": dur, "preferred_time": pt})
            seen.add(key)
        self._draft_path(d).write_text(json.dumps(draft, indent=2), encoding="utf-8")
        return draft

    def clear_draft(self, day: Optional[datetime] = None):
        d = (day or datetime.now(self.tz)).date().isoformat()
        p = self._draft_path(d);
        if p.exists(): p.unlink(missing_ok=True)

    # ---- Crystallization + reminders ----
    def crystallize_schedule(self, tasks: List[Dict], date: Optional[datetime] = None,
                             tz: Optional[str] = None, send_emails: bool = True) -> Dict[str, Any]:
        tz = ZoneInfo(tz) if tz else self.tz
        date = date or datetime.now(tz); day_str = date.date().isoformat()
        # pack sequentially, respecting preferred_time if provided
        cur_time = datetime.combine(date.date(), datetime.min.time()).replace(tzinfo=tz, hour=9, minute=0)
        events = []
        for t in tasks or []:
            start = cur_time
            if t.get("preferred_time"):
                try:
                    hh, mm = map(int, t["preferred_time"].split(":"))
                    start = start.replace(hour=hh, minute=mm)
                except Exception: pass
            duration = int(t.get("duration_min") or 30); end = start + timedelta(minutes=duration)
            events.append({"title": t.get("title","Task"), "start_local": start.isoformat(),
                           "end_local": end.isoformat(), "duration_min": duration})
            cur_time = end + timedelta(minutes=5)
        # persist
        day_file = self.daily_dir / f"{day_str}.json"
        day_file.write_text(json.dumps({"date": day_str, "tz": str(tz), "events": events}, indent=2), encoding="utf-8")
        # queue reminders at task start
        if send_emails and events:
            to_append = []
            for e in events:
                start_local = datetime.fromisoformat(e["start_local"]); start_utc = start_local.astimezone(timezone.utc)
                subj = f"[Schedule] {e['title']} — starts now"
                body = f"Task: {e['title']}\nStart (local {tz}): {start_local.strftime('%Y-%m-%d %H:%M')}\nDuration: {e['duration_min']} min\n"
                to_append.append(json.dumps({"subject": subj, "body": body, "send_at_utc": start_utc.isoformat(),
                                             "sent": False}, ensure_ascii=False))
            with self.reminders_file.open("a", encoding="utf-8") as f:
                f.write("\n".join(to_append) + ("\n" if to_append else ""))
            self.reminders.tick()
        return {"date": day_str, "events": events, "reminders_file": str(self.reminders_file)}

    # convenience: finalize from draft
    def finalize_draft(self, day: Optional[datetime] = None, tz: Optional[str] = None, send_emails: bool = True):
        d = (day or datetime.now(self.tz)).date().isoformat()
        draft = self.get_draft(day); out = self.crystallize_schedule(draft.get("tasks", []), date=datetime.fromisoformat(d).replace(tzinfo=self.tz), tz=tz, send_emails=send_emails)
        self.clear_draft(day); return out
