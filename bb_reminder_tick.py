# scripts/bb_reminder_tick.py
from __future__ import annotations

import os, json, base64, argparse
from pathlib import Path
from datetime import datetime, timezone
from email.message import EmailMessage

# Optional .env support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Gmail API
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

def _now_utc():
    return datetime.now(timezone.utc)

def _iso_to_dt_utc(s: str) -> datetime | None:
    try:
        # allow trailing 'Z'
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError:
                # tolerate UTF-8 BOM or minor hiccups
                try:
                    out.append(json.loads(s.encode("utf-8").decode("utf-8-sig")))
                except Exception:
                    continue
    return out

def _atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)

def _load_gmail_service(credentials_file: Path | None, token_file: Path | None):
    creds = None
    if token_file and token_file.exists():
        # accept token.json or token.pickle
        if token_file.suffix.lower() == ".pickle":
            import pickle
            with token_file.open("rb") as f:
                creds = pickle.load(f)
        else:
            creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        elif credentials_file and credentials_file.exists():
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_file), SCOPES)
            creds = flow.run_local_server(port=0)
        else:
            raise RuntimeError(
                "No valid Gmail OAuth credentials. "
                "Set GMAIL_CREDENTIALS_FILE and GMAIL_TOKEN_FILE or place credentials.json/token.json in the repo root."
            )
        # persist the refreshed/new token
        if token_file:
            if token_file.suffix.lower() == ".pickle":
                import pickle
                with token_file.open("wb") as f:
                    pickle.dump(creds, f)
            else:
                with token_file.open("w", encoding="utf-8") as f:
                    f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds, cache_discovery=False)

def _gmail_send(service, from_addr: str, to_addr: str, subject: str, body: str) -> dict:
    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body or "")
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    return service.users().messages().send(userId="me", body={"raw": raw}).execute()

def tick(reminders_path: Path, service, from_addr: str, to_addr: str, *, dry_run=False, max_send=50) -> dict:
    items = _read_jsonl(reminders_path)
    if not items:
        return {"sent": 0, "pending": 0, "reminders_file": str(reminders_path)}

    now = _now_utc()
    sent = 0
    for it in items:
        if it.get("sent") is True:
            continue
        s = it.get("send_at_utc")
        due_at = _iso_to_dt_utc(s) if isinstance(s, str) else None
        if not due_at or now < due_at:
            continue

        # due → attempt
        it["sent_attempted_at_utc"] = now.isoformat()
        subj = it.get("subject", "Reminder")
        body = it.get("body", "")
        if dry_run:
            it["sent"] = False
            it["error"] = "dry_run"
            sent += 1
        else:
            try:
                resp = _gmail_send(service, from_addr, to_addr, subj, body)
                it["sent"] = True
                it["sent_at_utc"] = _now_utc().isoformat()
                it["gmail_id"] = resp.get("id")
                sent += 1
            except HttpError as e:
                it["sent"] = False
                it["error"] = f"HttpError: {e}"
            except Exception as e:
                it["sent"] = False
                it["error"] = str(e)

        if sent >= max_send:
            break

    _atomic_write_jsonl(reminders_path, items)

    # recompute pending that are already due but not sent
    pending = 0
    now2 = _now_utc()
    for it in items:
        s = it.get("send_at_utc")
        if it.get("sent") is True or not isinstance(s, str):
            continue
        dt = _iso_to_dt_utc(s)
        if dt and now2 >= dt:
            pending += 1

    return {"sent": sent, "pending": pending, "reminders_file": str(reminders_path)}

def _resolve_planner_dir(cli_planner_dir: str | None) -> Path:
    if cli_planner_dir:
        return Path(cli_planner_dir)
    # Prefer centralized config if present
    try:
        from data_config import PLANNER_DIR  # project’s config
        return Path(str(PLANNER_DIR))
    except Exception:
        pass
    env = os.getenv("PLANNER_DIR")
    if env:
        return Path(env)
    # Fallback to repo-relative default
    return Path("data/planner")

def main() -> int:
    ap = argparse.ArgumentParser(description="Send due reminders from planner/reminders.jsonl via Gmail OAuth.")
    ap.add_argument("--planner-dir", default=None, help="Override planner dir (defaults to data_config.PLANNER_DIR or data/planner)")
    ap.add_argument("--dry-run", action="store_true", help="Do not send, only mark attempted")
    ap.add_argument("--max-send", type=int, default=50)
    ap.add_argument("--from", dest="from_addr", default=os.getenv("GMAIL_FROM"))
    ap.add_argument("--to", dest="to_addr", default=os.getenv("GMAIL_TO", os.getenv("SMTP_TO")))
    ap.add_argument("--token", dest="token_file", default=os.getenv("GMAIL_TOKEN_FILE", "token.json"))
    ap.add_argument("--creds", dest="creds_file", default=os.getenv("GMAIL_CREDENTIALS_FILE", "credentials.json"))
    args = ap.parse_args()

    planner_dir = _resolve_planner_dir(args.planner_dir)
    reminders_path = planner_dir / "reminders.jsonl"
    if not reminders_path.exists():
        print(f"[bb_reminder_tick] no reminders file at {reminders_path}")
        return 0

    if not args.from_addr or not args.to_addr:
        print("[bb_reminder_tick] missing GMAIL_FROM or GMAIL_TO")
        return 2

    try:
        svc = _load_gmail_service(Path(args.creds_file), Path(args.token_file))
    except Exception as e:
        print(f"[bb_reminder_tick] gmail auth failed: {e}")
        return 3

    out = tick(reminders_path, svc, args.from_addr, args.to_addr, dry_run=args.dry_run, max_send=args.max_send)
    print(f"[bb_reminder_tick] sent={out['sent']} pending={out['pending']} file={out['reminders_file']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
