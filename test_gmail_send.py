# test_gmail_send.py
import os
from assistant.behavioral_engine.schedulers.schedule_bridge import ScheduleBridge

# Ensure env is loaded (your app already calls dotenv.load_dotenv())
os.environ.setdefault("BB_MAIL_MODE", "gmail_api")
os.environ.setdefault("GMAIL_CREDENTIALS", r"C:\BiggerBrother-minimal\secrets\credentials.json")
os.environ.setdefault("GMAIL_TOKEN", r"C:\BiggerBrother-minimal\secrets\token.json")
os.environ.setdefault("SMTP_FROM", "you@example.com")
os.environ.setdefault("SMTP_TO", "you@example.com")

sb = ScheduleBridge(planner_dir=r"C:\BiggerBrother-minimal\data\planner")
sb.notifier.send("[Schedule] Hello from Gmail OAuth", "If you can read this, OAuth mail works.")
