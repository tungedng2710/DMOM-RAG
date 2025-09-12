from __future__ import annotations

import time
from typing import Iterable, List, Optional
import requests

from .config import settings


class Embeddings:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class OllamaEmbeddings(Embeddings):
    """Calls Ollama native embeddings endpoint.

    POST {base}/api/embeddings
    body: {"model": <embed_model>, "prompt": <text>}
    returns: {"embedding": [..]}
    """

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, timeout: int = 120):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.embedding_model
        self.timeout = timeout

    def _embed_one(self, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        resp = requests.post(url, json={"model": self.model, "prompt": text}, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            out.append(self._embed_one(t))
            # small sleep to avoid hammering local server
            time.sleep(0.01)
        return out


class SentenceTransformerEmbeddings(Embeddings):
    """Optional fallback using sentence-transformers if available locally.
    Not used by default because it requires downloading a model.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers not installed. Install it or use Ollama embeddings."
            ) from e
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


def get_default_embeddings() -> Embeddings:
    """Return the default embeddings client (Ollama)."""
    return OllamaEmbeddings()

