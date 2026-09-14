from __future__ import annotations

from typing import Dict, List, Optional

import requests

from .config import settings


class VLLMChat:
    """OpenAI-compatible chat client for the dedicated GLM 5.3 vLLM server."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 300,
    ):
        self.base_url = (base_url or settings.vllm_base_url).rstrip("/")
        self.model = model or settings.vllm_model
        self.api_key = (api_key or settings.vllm_api_key).strip()
        self.timeout = timeout

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        system: Optional[str] = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("VLLM_API_KEY is not configured")

        conversation: List[Dict[str, str]] = []
        if system:
            conversation.append({"role": "system", "content": system})
        conversation.extend(messages)

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": conversation,
                "temperature": max(0.0, float(temperature)),
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json() or {}
        choices = data.get("choices") or []
        if not choices:
            return ""
        content = (choices[0].get("message") or {}).get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "\n".join(
                part.get("text", "").strip()
                for part in content
                if isinstance(part, dict) and part.get("text", "").strip()
            )
        return ""


def get_default_chat() -> VLLMChat:
    """Return the single supported GLM 5.3 generation client."""

    return VLLMChat()
