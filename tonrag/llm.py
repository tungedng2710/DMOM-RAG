from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING
import requests

from .config import settings

try:
    from cerebras.cloud.sdk import Cerebras  # type: ignore
except Exception:
    Cerebras = None  # type: ignore

if TYPE_CHECKING:
    from cerebras.cloud.sdk import Cerebras as _CerebrasType  # type: ignore


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
    """Gemini chat via official google-genai client, with REST fallback.

    Primary path uses `google.genai` like in scripts/gemini_api_examples.py:
      - `Client(api_key=...)`
      - `client.models.generate_content(model=..., contents=...)`

    If the SDK is unavailable, falls back to the public REST endpoint
    `https://generativelanguage.googleapis.com/v1/models/{model}:generateContent`.
    """

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, timeout: int = 300, api_key: Optional[str] = None):
        self.base_url = (base_url or settings.gemini_base_url).rstrip("/")
        self.model = model or getattr(settings, "gemini_model", None) or settings.generation_model
        self.timeout = timeout

        # Resolve API key: prefer explicit override, else settings
        self.api_key = (api_key or getattr(settings, "gemini_api_key", None) or "").strip()

        # Try to initialize google-genai client; if unavailable, use REST fallback
        self._use_sdk = False
        self._sdk = None  # type: ignore
        if self.api_key:
            try:
                from google import genai  # type: ignore
                self._sdk = genai.Client(api_key=self.api_key)
                self._use_sdk = True
            except Exception:
                # Keep REST fallback
                self._use_sdk = False

    def _to_rest_contents(self, messages: List[Dict[str, str]]) -> List[Dict]:
        contents: List[Dict] = []
        for m in messages or []:
            role = (m.get("role") or "user").lower()
            text = (m.get("content") or "").strip()
            if not text:
                continue
            if role == "assistant":
                role = "model"
            elif role not in ("user", "model"):
                role = "user"
            contents.append({"role": role, "parts": [{"text": text}]})
        return contents

    def _to_sdk_contents(self, messages: List[Dict[str, str]]):
        # Build google.genai types.Content list preserving roles
        from google.genai import types  # type: ignore
        out = []
        for m in messages or []:
            role = (m.get("role") or "user").lower()
            text = (m.get("content") or "").strip()
            if not text:
                continue
            if role == "assistant":
                role = "model"
            elif role not in ("user", "model"):
                role = "user"
            out.append(types.Content(role=role, parts=[types.Part.from_text(text)]))
        return out

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2, system: Optional[str] = None) -> str:
        api_key = self.api_key or getattr(settings, "gemini_api_key", None)
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not configured")

        # Prefer SDK path if available
        if self._use_sdk and self._sdk is not None:
            try:
                from google.genai import types  # type: ignore
                contents = self._to_sdk_contents(messages)
                # If no messages were provided, nothing to send
                if not contents:
                    return ""
                config = types.GenerateContentConfig(
                    temperature=max(0.0, float(temperature)),
                    # Field name uses camelCase per SDK definition
                    systemInstruction=system if system else None,
                )
                resp = self._sdk.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                # SDK response has a convenient .text aggregator
                text = getattr(resp, "text", None)
                if isinstance(text, str) and text.strip():
                    return text.strip()
                # Fallback: attempt to join parts
                try:
                    cands = getattr(resp, "candidates", None) or []
                    for c in cands:
                        content = getattr(c, "content", None)
                        parts = getattr(content, "parts", None) or []
                        buf: List[str] = []
                        for p in parts:
                            t = getattr(p, "text", None)
                            if t:
                                buf.append(str(t).strip())
                        if buf:
                            return "\n".join(buf).strip()
                except Exception:
                    pass
                return ""
            except Exception:
                # On any SDK error, fall back to REST path below
                pass

        # REST fallback using public endpoint
        url = f"{self.base_url}/models/{self.model}:generateContent"
        payload: Dict = {
            "contents": self._to_rest_contents(messages),
            "generationConfig": {"temperature": max(0.0, float(temperature))},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        resp = requests.post(url, params={"key": api_key}, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json() or {}
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


class CerebrasChat:
    """Cerebras Cloud chat wrapper."""

    def __init__(self, model: Optional[str] = None, timeout: int = 300, api_key: Optional[str] = None):
        self.model = model or getattr(settings, "cerebras_model", None) or settings.generation_model
        self.timeout = timeout
        self.api_key = (api_key or getattr(settings, "cerebras_api_key", None) or "").strip()
        self._client: Optional["_CerebrasType"] = None

    def _ensure_client(self) -> "_CerebrasType":
        if Cerebras is None:
            raise RuntimeError(
                "cerebras_cloud_sdk is not installed. Install with `pip install cerebras_cloud_sdk`."
            )
        if not self.api_key:
            raise RuntimeError("CEREBRAS_API_KEY not configured")
        if self._client is None:
            self._client = Cerebras(api_key=self.api_key)
        return self._client

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2, system: Optional[str] = None) -> str:
        # Normalize temperature to valid range
        temperature = max(0.0, float(temperature))

        conversation: List[Dict[str, str]] = []
        if system:
            conversation.append({"role": "system", "content": system})
        for msg in messages or []:
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            role = msg.get("role") or "user"
            conversation.append({"role": role, "content": content})
        if not conversation:
            return ""

        client = self._ensure_client()
        response = client.chat.completions.create(
            messages=conversation,
            model=self.model,
            temperature=temperature,
        )
        try:
            choice = response.choices[0]  # type: ignore[index]
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts: List[str] = []
                for part in content:
                    if isinstance(part, str):
                        if part.strip():
                            parts.append(part.strip())
                    elif isinstance(part, dict):
                        text = part.get("text")  # type: ignore[call-arg]
                        if isinstance(text, str) and text.strip():
                            parts.append(text.strip())
                if parts:
                    return "\n".join(parts).strip()
        except Exception:
            pass
        return str(response)

def get_default_chat(backend: Optional[str] = None, *, api_key: Optional[str] = None):
    """Return chat client based on backend preference or env.

    backend: 'ollama' | 'gemini' | 'cerebras' | None
      - None: derive from settings.chat_backend (defaults to ollama)
    """
    choice = (backend or getattr(settings, 'chat_backend', None) or 'ollama').lower()
    if choice.startswith('gemini'):
        return GeminiChat(api_key=api_key)
    if choice.startswith('cerebras'):
        return CerebrasChat(api_key=api_key)
    # default to Ollama
    return OllamaChat()
