
import os, json, io
from pathlib import Path
from datetime import datetime
import pytest

jsonschema = pytest.importorskip("jsonschema")
from jsonschema import Draft202012Validator

def load_schema():
    # Allow override via env; else assume repo layout ../schemas/extraction_record.schema.json
    schema_env = os.getenv("EXTRACTION_SCHEMA")
    if schema_env:
        p = Path(schema_env)
    else:
        p = Path(__file__).resolve().parents[1] / "BiggerBrother-minimal" / "schemas" / "extraction_record.schema.json"
    assert p.exists(), f"Schema not found at {p}"
    with p.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    Draft202012Validator.check_schema(schema)
    return schema

def test_extraction_schema_happy_path(tmp_path):
    schema = load_schema()
    validator = Draft202012Validator(schema)

    sample = {
        "entries": [
            {"category":"meals","data":{"what":"banquet meal","calories":410},"confidence":0.86},
            {"category":"sleep","data":{"bed":"02:10","wake":"14:00"}}
        ],
        "schedule_suggestions": [
            {"title":"Therapy travel buffer","duration_min":30,"preferred_time":"08:00","day":"today"},
            {"title":"Meal prep","duration_min":25,"preferred_time":None,"day":"2025-09-06"}
        ],
        "triggers":{"planning":"start"},
        "load_context":[{"category":"meals","days_back":7,"max_entries":50}],
        "inventory_updates":[{"category":"inventory_pantry","item":"banquet meal","qty":-1,"unit":"tray"}]
    }
    validator.validate(sample)  # Raises on failure

def test_schedule_bridge_draft_and_finalize(tmp_path, monkeypatch):
    sb_mod = pytest.importorskip("assistant.behavioral_engine.schedulers.schedule_bridge", reason="Scheduler Bridge not installed yet")
    ScheduleBridge = sb_mod.ScheduleBridge

    # Ensure predictable timezone for this test
    monkeypatch.setenv("BB_TZ", "America/New_York")

    bridge = ScheduleBridge(planner_dir=str(tmp_path), openai_client=None)

    # Fixed day for determinism
    import zoneinfo
    from datetime import datetime
    tz = zoneinfo.ZoneInfo("America/New_York")
    day = datetime(2025, 9, 6, 9, 0, tzinfo=tz)

    # Append suggestions (including a duplicate)
    suggestions = [
        {"title":"Finish v0.1 packaging","duration_min":90,"preferred_time":"11:00"},
        {"title":"Therapy prep notes","duration_min":30,"preferred_time":"13:30"},
        {"title":"Finish v0.1 packaging","duration_min":90,"preferred_time":"11:00"}  # dup
    ]
    draft = bridge.append_suggestions(suggestions, day=day)
    assert len(draft["tasks"]) == 2, f"De-dup failed: tasks={draft['tasks']}"

    # Finalize and check artifacts
    out = bridge.crystallize_schedule(draft["tasks"], date=day, tz="America/New_York", send_emails=True)
    daily_path = Path(tmp_path) / "daily" / f"{out['date']}.json"
    assert daily_path.exists(), "Daily schedule JSON not written"

    # Verify reminders were queued (emails won't send without SMTP env; that's fine)
    reminders_path = Path(tmp_path) / "reminders.jsonl"
    assert reminders_path.exists(), "Reminders file missing"
    lines = [json.loads(l) for l in reminders_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == len(out["events"]), "One reminder per event expected"

    # Check that send_at_utc is synchronized with each event's local start time
    from datetime import datetime as dt, timezone
    for ev, rem in zip(out["events"], lines):
        start_local = dt.fromisoformat(ev["start_local"])
        start_utc = start_local.astimezone(timezone.utc).isoformat()
        assert rem["send_at_utc"].startswith(start_utc[:16]), "Reminder time should match event start (to the minute)"
