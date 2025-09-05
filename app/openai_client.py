# app/openai_client.py
from __future__ import annotations

import os, json, typing as t
import sys
from datetime import datetime, timezone, time
import hashlib

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

def _offline_reason(self) -> str | None:
    if os.getenv("AGENT_OFFLINE") == "1":
        return "AGENT_OFFLINE=1"
    try:
        import openai  # noqa
    except Exception:
        return "openai SDK not importable"
    if not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY missing"
    return None

def _offline(self) -> bool:
    return self._offline_reason() is not None

def _safe(obj):
    # Best effort to stringify OpenAI SDK responses without exploding
    try:
        # v1 client objects often have model_dump_json/model_dump
        if hasattr(obj, "model_dump_json"):
            return obj.model_dump_json(indent=2, ensure_ascii=False)
        if hasattr(obj, "model_dump"):
            return json.dumps(obj.model_dump(mode="json"), indent=2, ensure_ascii=False)
    except Exception:
        pass
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)

def _dbg(label: str, payload):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    #if isinstance(payload, (dict, list)):
        #print(f"[API-DEBUG {ts}] {label}:\n{_safe(payload)}", file=sys.stderr, flush=True)
    #else:
        #print(f"[API-DEBUG {ts}] {label}: {payload}", file=sys.stderr, flush=True)

class OpenAIClient:
    """
    .complete(prompt, model=None, **kwargs) -> str
    - Chat models -> chat.completions
    - Legacy text models -> completions
    - Offline (no key / AGENT_OFFLINE=1 / import fail) -> deterministic JSON
    - Never raises; returns '[ERROR COMPLETION: ...]' on exceptions
    """
    def __init__(self, raise_on_fail: bool | None = None) -> None:
        self.raise_on_fail = True if raise_on_fail is None else raise_on_fail
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

    def chat(self, *, messages, model: str, **kwargs) -> str:
        reason = _offline_reason(self)
        # Print request upfront
        _dbg("REQUEST.chat", {
            "model": model,
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            # DO NOT log headers or keys
            "messages": messages,
            "kwargs": {k: v for k, v in kwargs.items() if k not in ("api_key", "headers")}
        })

        if reason:
            _dbg("OFFLINE", reason)
            # your existing stub path
            stub = _offline_payload(messages[-1]["content"] if messages else "")
            _dbg("RESPONSE.chat (offline stub)", stub)
            return stub

        t0 = datetime.now(timezone.utc)
        try:
            # Example using the modern client; adapt to whatever you’re using:
            # resp = self._sdk.chat.completions.create(model=model, messages=messages, **kwargs)
            resp = self._sdk.chat.completions.create(model=model, messages=messages, **kwargs)

            dt = datetime.now(timezone.utc) - t0
            elapsed = dt.total_seconds()
            _dbg("RESPONSE.chat.raw", {"elapsed_s": round(elapsed, 3), "resp": resp})

            # Extract text content
            content = None
            try:
                content = resp.choices[0].message.content
            except Exception:
                # Fallback for Responses API or other shapes
                content = getattr(resp, "output_text", None) or str(resp)

            _dbg("RESPONSE.chat.content", content)
            return content

        except Exception as e:
            # OpenAI Python v1 exceptions usually have status_code + response
            status = getattr(e, "status_code", None)
            body = None
            try:
                r = getattr(e, "response", None)
                if r is not None:
                    # requests-like
                    body = getattr(r, "text", None) or getattr(r, "content", None)
            except Exception:
                body = None

            _dbg("ERROR.chat", {
                "type": type(e).__name__,
                "status": status,
                "message": str(e),
                "body": body[:2000] if isinstance(body, (str, bytes)) else body
            })

            # Special attention for 4xx as requested
            if status and 400 <= status < 500:
                _dbg("ERROR.chat.4xx", {"status": status, "body": body})

            if self.raise_on_fail:
                raise
            return "__OPENAI_CALL_FAILED__"

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

    # --- Embeddings (new) ---------------------------------------------------
    def embed(self, inputs: t.Union[str, list[str]], model: str | None = None, dim: int = 1536) -> list[list[float]]:
        """
        Return embeddings for one or many strings.
        - Uses OpenAI v1 client if available, else old module, else deterministic offline stub.
        - Always returns List[List[float]] (even for a single input).
        """
        model = model or os.getenv("OPENAI_EMBED_MODEL") or "text-embedding-3-large"
        texts: list[str] = [inputs] if isinstance(inputs, str) else list(inputs)

        _dbg("REQUEST.embed", {
            "model": model,
            "count": len(texts),
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        })

        reason = _offline_reason(self)
        if reason:
            _dbg("OFFLINE", reason)
            return [self._offline_embed(t, dim=dim) for t in texts]

        try:
            # Prefer modern client if present
            if self._new and hasattr(self.client, "embeddings"):
                resp = self.client.embeddings.create(model=model, input=texts)
                vecs = [d.embedding for d in resp.data]
            else:
                # Old 0.x style
                resp = self._sdk.Embedding.create(model=model, input=texts)  # type: ignore[attr-defined]
                vecs = [d["embedding"] for d in resp["data"]]
            _dbg("RESPONSE.embed", {"count": len(vecs)})
            return vecs
        except Exception as e:
            _dbg("ERROR.embed", {"type": type(e).__name__, "message": str(e)})
            if self.raise_on_fail:
                raise
            # graceful fallback
            return [self._offline_embed(t, dim=dim) for t in texts]

    def _offline_embed(self, text: str, *, dim: int = 1536) -> list[float]:
        """
        Deterministic, normalized vector for offline/testing paths.
        """
        seed = int.from_bytes(hashlib.sha1((text or "").encode("utf-8")).digest()[:8], "big")
        # simple Weyl sequence  mod to fake Gaussian-ish spread
        x, a, m = seed % 2147483647 or 1, 1103515245, 2**31 - 1
        vals = []
        ssum = 0.0
        for _ in range(dim):
            x = (a * x + 12345) % m
            # scale to [-1,1]
            v = ((x / m) * 2.0) - 1.0
            vals.append(v)
            ssum = v * v
        n = (ssum ** 0.5) or 1.0
        return [v / n for v in vals]

def _looks_like_chat_model(model: str) -> bool:
    m = model.lower()
    return ("gpt-" in m) or ("-turbo" in m) or ("-o" in m)

# one shared instance
default_client = OpenAIClient()
