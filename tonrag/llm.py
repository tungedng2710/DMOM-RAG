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


def get_default_chat() -> OllamaChat:
    return OllamaChat()

