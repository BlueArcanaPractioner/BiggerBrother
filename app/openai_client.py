# app/openai_client.py
from __future__ import annotations
import os, json, typing as t
from dotenv import load_dotenv

load_dotenv()

# --- model/kwargs policy -----------------------------------------
MODEL_CHAT_ONLY = ("gpt-5-nano",)  # extend as needed

DEFAULT_CHAT_KWARGS = {"response_format"}  # nano: no temp/top_p/max_tokens
DEFAULT_FULL_KWARGS = {"response_format", "temperature", "top_p", "max_tokens"}

def _allowlist_for(model: str) -> set[str]:
    m = (model or "").lower()
    if any(m.startswith(x) for x in MODEL_CHAT_ONLY):
        return DEFAULT_CHAT_KWARGS
    return DEFAULT_FULL_KWARGS

def _offline_payload(prompt: str) -> str:
    """Deterministic, schema-shaped reply for offline/testing."""
    return json.dumps({
        "topic": [{"label": "auto", "p": 1.0}],
        "tone": [],
        "intent": [],
        "confidence": 0.5,
    })

class OpenAIClient:
    """
    .complete(prompt, model=None, **kwargs) -> str
    - Chat models -> chat.completions
    - Legacy text models -> completions
    - Offline (no key / AGENT_OFFLINE=1 / import fail) -> deterministic JSON
    - Never raises; returns '[ERROR COMPLETION: ...]' on exceptions
    """
    def __init__(self) -> None:
        self._sdk = None
        self._new = False
        self.client = None
        try:
            import openai as _sdk  # old 0.x API *module*
            self._sdk = _sdk
            try:
                # new 1.x API (client object)
                from openai import OpenAI  # type: ignore
                self.client = OpenAI()
                self._new = True
            except Exception:
                self.client = _sdk  # fall back to old module-style
        except Exception:
            self._sdk = None
            self.client = None

    def _offline(self) -> bool:
        return (
            os.getenv("AGENT_OFFLINE") == "1"
            or self._sdk is None
            or not os.getenv("OPENAI_API_KEY")
        )

    def chat(self, messages: list[dict], model: str | None = None, **kwargs) -> str:
        m = model or os.getenv("OPENAI_MODEL") or "gpt-5-nano"
        if self._offline():
            # reuse your offline payload helper
            return _offline_payload(messages[-1]["content"] if messages else "")
        kw = {k: v for k, v in kwargs.items() if k in _allowlist_for(m)}
        try:
            if self._new and hasattr(self.client, "chat"):
                resp = self.client.chat.completions.create(model=m, messages=messages, **kw)
                return resp.choices[0].message.content or ""
            # old 0.x fallback
            resp = self.client.ChatCompletion.create(model=m, messages=messages, **kw)  # type: ignore[attr-defined]
            return resp["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[ERROR CHAT: {e}]"

    def complete(self, prompt: str, model: str | None = None, **kwargs) -> str:
        # legacy text-* path, kept for completeness
        m = model or os.getenv("OPENAI_MODEL") or "text-davinci-003"
        if self._offline():
            return _offline_payload(prompt)
        kw = {k: v for k, v in kwargs.items() if k in DEFAULT_FULL_KWARGS}
        try:
            if self._new and hasattr(self.client, "completions"):
                resp = self.client.completions.create(model=m, prompt=prompt, **kw)
                return resp.choices[0].text or ""
            resp = self.client.Completion.create(model=m, prompt=prompt, **kw)  # type: ignore[attr-defined]
            return resp["choices"][0]["text"]
        except Exception as e:
            return f"[ERROR COMPLETION: {e}]"

def _looks_like_chat_model(model: str) -> bool:
    m = model.lower()
    return ("gpt-" in m) or ("-turbo" in m) or ("-o" in m)

# one shared instance
default_client = OpenAIClient()
