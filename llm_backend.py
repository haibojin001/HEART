from __future__ import annotations

import abc
import json
import os
import re
from typing import Any, Dict, List, Optional


class LLMBackend(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
    ) -> str: ...

    _FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

    def chat_json(self, messages: List[Dict[str, str]], **kw) -> Dict[str, Any]:
        raw = self.chat(messages, **kw)
        try:
            return json.loads(raw)
        except Exception:
            pass
        m = self._FENCE_RE.search(raw)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        s, e = raw.find("{"), raw.rfind("}")
        if 0 <= s < e:
            try:
                return json.loads(raw[s : e + 1])
            except Exception:
                pass
        raise ValueError(f"Backend returned non-JSON response:\n{raw[:600]}")


class OpenAIBackend(LLMBackend):

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
    ) -> str:
        import requests
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop
        r = requests.post(f"{self.base_url}/chat/completions",
                          headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class AnthropicBackend(LLMBackend):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
    ) -> str:
        import requests
        sys_prompt = ""
        msgs = []
        for m in messages:
            if m["role"] == "system":
                sys_prompt = (sys_prompt + "\n\n" + m["content"]).strip()
            else:
                msgs.append({"role": m["role"], "content": m["content"]})
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": msgs,
        }
        if sys_prompt:
            payload["system"] = sys_prompt
        if stop:
            payload["stop_sequences"] = stop
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


class MockBackend(LLMBackend):

    name = "mock"

    _ROLE_RE = re.compile(
        r"You are (?:the|a) (Planner|Router|Tool Primitive|Verifier)", re.IGNORECASE,
    )

    def __init__(self):
        self.handlers: Dict[str, List[tuple]] = {}
        self.default_response = json.dumps({"status": "MOCK_UNHANDLED"})

    def register(self, role: str, pattern: str, response) -> None:
        role_norm = self._normalize_role(role)
        self.handlers.setdefault(role_norm, []).append(
            (re.compile(pattern, re.DOTALL | re.IGNORECASE), response)
        )

    @staticmethod
    def _normalize_role(s: str) -> str:
        s = s.strip().lower()
        if "planner" in s: return "planner"
        if "router" in s: return "router"
        if "primitive" in s: return "tool primitive"
        if "verifier" in s: return "verifier"
        return s

    def _detect_role(self, sys_text: str) -> str:
        m = self._ROLE_RE.search(sys_text)
        if m:
            return self._normalize_role(m.group(1))
        low = sys_text.lower()
        for cand in ("planner", "router", "tool primitive", "verifier"):
            if cand in low:
                return cand
        return "unknown"

    def chat(self, messages, temperature=0.2, max_tokens=1024, stop=None) -> str:
        sys_text = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user_text = "\n".join(m["content"] for m in messages if m["role"] == "user")
        role = self._detect_role(sys_text)
        for pat, resp in self.handlers.get(role, []):
            if pat.search(user_text):
                return resp(user_text) if callable(resp) else resp
        return self.default_response
