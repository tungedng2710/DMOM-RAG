from __future__ import annotations

from typing import Dict, List, Optional
import requests

from .config import settings


class OllamaChat:
    """Minimal wrapper around Ollama /api/chat with non-streaming responses."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, timeout: int = 300):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.generation_model
        self.timeout = timeout

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2, system: Optional[str] = None) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": ([] if system is None else [{"role": "system", "content": system}]) + messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        # Expected shape: { message: {role, content}, ... }
        msg = data.get("message") or {}
        return (msg.get("content") or "").strip()


class GeminiChat:
    """Minimal wrapper around Google Gemini generateContent API (non-streaming).

    Requires environment variables in `.env` (see config Settings):
    - `GEMINI_API_KEY`
    - `GEMINI_MODEL` (e.g., `gemini-1.5-flash` or `gemini-1.5-pro`)
    Optionally override base URL via `GEMINI_BASE_URL` (defaults to v1 endpoint).
    """

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, timeout: int = 300):
        self.base_url = (base_url or settings.gemini_base_url).rstrip("/")
        # Prefer explicit Gemini model if provided in settings
        self.model = model or getattr(settings, "gemini_model", None) or settings.generation_model
        self.timeout = timeout

    def _to_contents(self, messages: List[Dict[str, str]]) -> List[Dict]:
        contents: List[Dict] = []
        for m in messages or []:
            role = (m.get("role") or "user").lower()
            text = (m.get("content") or "").strip()
            if not text:
                continue
            # Gemini expects roles: "user" or "model"
            if role == "assistant":
                role = "model"
            elif role not in ("user", "model"):
                role = "user"
            contents.append({
                "role": role,
                "parts": [{"text": text}],
            })
        return contents

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2, system: Optional[str] = None) -> str:
        api_key = getattr(settings, "gemini_api_key", None)
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not configured in environment")

        url = f"{self.base_url}/models/{self.model}:generateContent"
        payload: Dict = {
            "contents": self._to_contents(messages),
            "generationConfig": {
                "temperature": max(0.0, float(temperature)),
            },
        }
        if system:
            payload["systemInstruction"] = {
                "parts": [{"text": system}],
            }

        resp = requests.post(
            url,
            params={"key": api_key},
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json() or {}

        # Expected shape (simplified):
        # {
        #   "candidates": [
        #     { "content": { "parts": [{"text": "..."}, ...] }, ... }, ...
        #   ]
        # }
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        first = candidates[0] or {}
        content = first.get("content") or {}
        parts = content.get("parts") or []
        texts = []
        for p in parts:
            t = (p.get("text") or "").strip()
            if t:
                texts.append(t)
        return "\n".join(texts).strip()

def get_default_chat(backend: Optional[str] = None):
    """Return chat client based on backend preference or env.

    backend: 'ollama' | 'gemini' | None
      - None: derive from settings.chat_backend (defaults to ollama)
    """
    choice = (backend or getattr(settings, 'chat_backend', None) or 'ollama').lower()
    if choice.startswith('gemini'):
        return GeminiChat()
    # default to Ollama
    return OllamaChat()
