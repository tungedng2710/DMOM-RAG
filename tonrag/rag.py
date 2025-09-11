from __future__ import annotations

from typing import Dict, List, Optional

from .config import settings
from .embeddings import get_default_embeddings
from .vectorstore import ChromaStore
from .llm import get_default_chat


SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using the provided context."
    " If the answer is not contained in the context, say you don't know. Be concise."
)


def build_prompt(question: str, contexts: List[str]) -> List[Dict[str, str]]:
    context_block = "\n\n".join([f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts)])
    user = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    return [{"role": "user", "content": user}]


class RAGPipeline:
    def __init__(self, top_k: Optional[int] = None):
        self.emb = get_default_embeddings()
        self.store = ChromaStore()
        self.chat = get_default_chat()
        self.top_k = top_k or settings.top_k

    def retrieve(self, query: str, top_k: Optional[int] = None):
        k = top_k or self.top_k
        q_emb = self.emb.embed_query(query)
        results = self.store.query(q_emb, top_k=k)
        return results

    def generate(self, question: str, retrieved: List[Dict]) -> str:
        contexts = [r["document"] for r in retrieved]
        messages = build_prompt(question, contexts)
        answer = self.chat.generate(messages, system=SYSTEM_PROMPT)
        return answer

    def answer(self, question: str, top_k: Optional[int] = None) -> Dict:
        hits = self.retrieve(question, top_k=top_k)
        answer = self.generate(question, hits)
        return {"answer": answer, "contexts": hits}

