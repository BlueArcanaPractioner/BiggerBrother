from __future__ import annotations
import os, json, uuid, smtplib, time, base64
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import List, Dict, Optional, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import dotenv

# --- Gmail API deps (OAuth) ---
from base64 import urlsafe_b64encode
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
except Exception:
    Credentials = None
    InstalledAppFlow = None
    Request = None
    build = None
dotenv.load_dotenv()

class EmailNotifier:
    def __init__(self, env: dict, default_tz: str = "America/New_York"):
        self.tz = ZoneInfo(env.get("BB_TZ", default_tz))
        self.smtp_host = env.get("SMTP_HOST"); self.smtp_port = int(env.get("SMTP_PORT", "587"))
        self.smtp_user = env.get("SMTP_USER"); self.smtp_pass = env.get("SMTP_PASS")
        self.from_addr = env.get("SMTP_FROM"); self.to_addr = env.get("SMTP_TO")
        self.enabled = all([self.smtp_host, self.smtp_user, self.smtp_pass, self.from_addr, self.to_addr])
        self.dry_run = env.get("BB_EMAIL_DRY_RUN", "0") == "1"

    def send(self, subject: str, body: str) -> bool:
        if not self.enabled or self.dry_run:
            print(f"[info] email {'dry-run' if self.dry_run else 'disabled'}: {subject}")
            return False
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

class GmailApiNotifier:
    """
    Minimal Gmail API sender using OAuth token.json.
    Requires scope: https://www.googleapis.com/auth/gmail.send
    Env:
      - GMAIL_CREDENTIALS_JSON: path to OAuth client secrets (credentials.json)
      - GMAIL_TOKEN_JSON:       path to token.json (created after first consent)
      - BB_EMAIL_TO or SMTP_TO: recipient (default 'me' if omitted)
      - BB_EMAIL_FROM: optional From header (Gmail will still send as the account)
      - BB_TZ: timezone (e.g., America/Kentucky/Louisville)
    """
    SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

    def __init__(self, env: dict):
        self.tz = ZoneInfo(env.get("BB_TZ", "America/New_York"))
        self.creds_path = Path(env.get("GMAIL_CREDENTIALS_JSON", "credentials.json"))
        self.token_path = Path(env.get("GMAIL_TOKEN_JSON", "token.json"))
        self.to_addr = env.get("BB_EMAIL_TO") or env.get("SMTP_TO")
        self.from_addr = env.get("BB_EMAIL_FROM")  # optional header
        self.enabled = bool(build) and self.creds_path.exists()
        self._service = None

    def _ensure_service(self) -> bool:
        if not self.enabled:
            print("[warn] GmailApiNotifier disabled: missing google-api libs or credentials.json")
            return False
        creds = None
        if self.token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(self.token_path), self.SCOPES)
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
            except Exception as e:
                print(f"[warn] token.json invalid, will re-auth: {e}")
                creds = None
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(str(self.creds_path), self.SCOPES)
            # Port 0 lets the OS pick a free port; override with BB_OAUTH_PORT if needed.
            port = int(os.getenv("BB_OAUTH_PORT", "0"))
            creds = flow.run_local_server(port=port)
            self.token_path.write_text(creds.to_json(), encoding="utf-8")
        self._service = build("gmail", "v1", credentials=creds)
        return True

    def send(self, subject: str, body: str) -> bool:
        if not self._ensure_service():
            return False
        # If no explicit recipient was provided, Gmail will send to the account if header is omitted.
        if not self.to_addr:
            self.to_addr = "me"
        msg = MIMEText(body, "plain", "utf-8")
        msg["To"] = self.to_addr
        if self.from_addr:
            msg["From"] = self.from_addr
        msg["Subject"] = subject
        raw = urlsafe_b64encode(msg.as_bytes()).decode("ascii")
        try:
            self._service.users().messages().send(userId="me", body={"raw": raw}).execute()
            return True
        except Exception as e:
            print(f"[warn] gmail send failed: {e}")
            return False

class GmailOAuthNotifier:
    """
    Gmail API sender using OAuth2 credentials.json -> token.(json|pickle).
    Env:
      GMAIL_CREDENTIALS : path to client credentials.json
      GMAIL_TOKEN       : path to token.json (or token.pickle for legacy)
      SMTP_FROM / SMTP_TO: addresses for From/To (reuse existing names)
      BB_TZ             : timezone (default America/New_York)
      BB_OAUTH_PORT     : local port for the first-time auth flow (default 8765)
    """
    SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

    def __init__(self, env: dict):
        self.tz = ZoneInfo(env.get("BB_TZ", "America/New_York"))
        self.credentials_path = env.get("GMAIL_CREDENTIALS", "credentials.json")
        self.token_path = env.get("GMAIL_TOKEN", "token.json")
        self.from_addr = env.get("SMTP_FROM") or env.get("MAIL_FROM")
        self.to_addr   = env.get("SMTP_TO")   or env.get("MAIL_TO")
        self.enabled = bool(self.from_addr and self.to_addr and self._ensure_creds())

    def _ensure_creds(self) -> bool:
        try:
            # Lazy imports so non-Gmail users don't need the packages
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
        except Exception as e:
            print(f"[warn] gmail oauth libs not available: {e}")
            return False

        creds = None
        # Load token.(json|pickle)
        if os.path.exists(self.token_path):
            try:
                if self.token_path.lower().endswith(".pickle"):
                    import pickle
                    with open(self.token_path, "rb") as f:
                        creds = pickle.load(f)
                else:
                    creds = Credentials.from_authorized_user_file(self.token_path, self.SCOPES)
            except Exception:
                creds = None

        if not creds or not creds.valid:
            if creds and creds.expired and getattr(creds, "refresh_token", None):
                try:
                    creds.refresh(Request())
                except Exception:
                    creds = None
            if not creds:
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, self.SCOPES)
                port = int(os.getenv("BB_OAUTH_PORT", "8765"))
                creds = flow.run_local_server(port=port, prompt="consent")
            # Persist token as JSON (modern quickstart style)
            try:
                with open(self.token_path, "w", encoding="utf-8") as token:
                    token.write(creds.to_json())
            except Exception:
                pass

        try:
            from googleapiclient.discovery import build
            # Disable on-disk discovery cache (works well in CLI/daemon)
            self.service = build("gmail", "v1", credentials=creds, cache_discovery=False)
            self.creds = creds
            return True
        except Exception as e:
            print(f"[warn] gmail service init failed: {e}")
            return False

    def send(self, subject: str, body: str) -> bool:
        if not self.enabled:
            return False
        try:
            msg = MIMEMultipart()
            msg["From"] = self.from_addr
            msg["To"] = self.to_addr
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain", "utf-8"))
            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
            self.service.users().messages().send(userId="me", body={"raw": raw}).execute()
            return True
        except Exception as e:
            print(f"[warn] gmail api send failed: {e}")
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

        env = (mail_env or os.environ)
        transport = (env.get("BB_MAIL_TRANSPORT") or "").lower()
        use_gmail = transport == "gmail" or env.get("GMAIL_TOKEN_JSON") or env.get("GMAIL_CREDENTIALS_JSON")
        if use_gmail and Credentials and build:
            self.notifier = GmailApiNotifier(env)
        else:
            self.notifier = EmailNotifier(env)
        self.reminders = ReminderDaemon(self.reminders_file, self.notifier)
        mode = (env.get("BB_MAIL_MODE", "smtp") or "smtp").lower()
        if mode == "gmail_api":
            self.notifier = GmailOAuthNotifier(env)
            if not self.notifier.enabled:
                print("[warn] gmail_api mode requested but not enabled; falling back to SMTP")
                self.notifier = EmailNotifier(env)
        else:
            self.notifier = EmailNotifier(env)
        self.reminders = ReminderDaemon(self.reminders_file, self.notifier)

        # Optional: small background pulse to deliver reminders at minute granularity.
        # Enable with BB_REMINDER_PULSE_SECS (e.g., 60)
        try:
            pulse = int(env.get("BB_REMINDER_PULSE_SECS", "0"))
        except Exception:
            pulse = 0
        if pulse > 0:
            def _loop():
                while True:
                    try:
                        self.reminders.tick()
                    except Exception as e:
                        print(f"[warn] reminders.tick: {e}")
                    time.sleep(pulse)
            threading.Thread(target=_loop, daemon=True).start()

    def status(self) -> dict:
        """Quick snapshot for debugging."""
        rem_count = 0
        if self.reminders_file.exists():
            try: rem_count = len([l for l in self.reminders_file.read_text(encoding="utf-8").splitlines() if l.strip()])
            except Exception: rem_count = -1
        return {
            "tz": str(self.tz),
            "emails_enabled": self.notifier.enabled,
            "email_dry_run": getattr(self.notifier, "dry_run", False),
            "reminders_file": str(self.reminders_file),
            "queued_reminders": rem_count,
            "outbox_dir": str(self.outbox_dir),
            "drafts_dir": str(self.drafts_dir),
            "daily_dir": str(self.daily_dir),
        }

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
        # always tick; notifier respects enabled/dry-run
        self.reminders.tick()
        # also write human-readable outbox artifacts for debugging or dry-run
        try:
            for i, e in enumerate(events, 1):
                start_local = datetime.fromisoformat(e["start_local"])
                slug = "".join(c for c in e["title"] if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
                out = self.outbox_dir / f"{day_str}_{i:02d}_{slug}.txt"
                out.write_text(
                    f"{e['title']}\nStart: {start_local.strftime('%Y-%m-%d %H:%M %Z')}\nDuration: {e['duration_min']} min\n",
                    encoding="utf-8"
                )
        except Exception as _:
            pass
    return {"date": day_str, "events": events, "reminders_file": str(self.reminders_file)}

# convenience: finalize from draft
def finalize_draft(self, day: Optional[datetime] = None, tz: Optional[str] = None, send_emails: bool = True):
    d = (day or datetime.now(self.tz)).date().isoformat()
    draft = self.get_draft(day); out = self.crystallize_schedule(draft.get("tasks", []), date=datetime.fromisoformat(d).replace(tzinfo=self.tz), tz=tz, send_emails=send_emails)
    self.clear_draft(day); return out
