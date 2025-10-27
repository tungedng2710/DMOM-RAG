from __future__ import annotations

from typing import Dict, List, Optional

from .config import settings
from .embeddings import get_default_embeddings
from .vectorstore import ChromaStore
from .llm import get_default_chat


SYSTEM_PROMPT = (
    "Bạn là bác sĩ trả lời câu hỏi y khoa bằng tiếng Việt, ngắn gọn và chính xác. "
    "Chỉ sử dụng thông tin trong ngữ cảnh. Nếu thiếu thông tin, hãy nói: "
    '"Không đủ thông tin từ nguồn để trả lời đầy đủ". '
    "Trích dẫn nguồn bằng số thứ tự trong ngoặc vuông như [1], [2] dựa trên ngữ cảnh."
)


def build_prompt(question: str, contexts: List[str]) -> List[Dict[str, str]]:
    # Number each context for [n] citations
    blocks = []
    for i, c in enumerate(contexts, 1):
        blocks.append(f"Nguồn [{i}]:\n{c}")
    context_block = "\n\n".join(blocks)
    user = (
        f"Ngữ cảnh:\n{context_block}\n\n"
        f"Câu hỏi: {question}\n\n"
        f"Hãy trả lời bằng tiếng Việt, có trích dẫn [n]."
    )
    return [{"role": "user", "content": user}]


class RAGPipeline:
    def __init__(
        self,
        top_k: Optional[int] = None,
        llm: Optional[str] = None,
        api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ):
        self.emb = get_default_embeddings()
        self.store = ChromaStore(create_if_missing=False)
        # Accept a generic API key override (Gemini legacy alias kept for compatibility)
        key_override = api_key or gemini_api_key
        self.chat = get_default_chat(llm, api_key=key_override)
        self.top_k = top_k or settings.top_k

    def retrieve(self, query: str, top_k: Optional[int] = None):
        k = top_k or self.top_k
        mode = (settings.chroma_query_mode or "auto").lower()
        if mode in ("text", "auto"):
            try:
                return self.store.query_text(query, top_k=k)
            except Exception:
                # Fall back to embeddings if text mode isn't supported
                pass
        if mode in ("embed", "auto"):
            q_emb = self.emb.embed_query(query)
            return self.store.query(q_emb, top_k=k)
        # Default fallback
        q_emb = self.emb.embed_query(query)
        return self.store.query(q_emb, top_k=k)

    def _parse_chunk(self, doc: str) -> Dict[str, str]:
        q = a = r = ""
        for line in (doc or "").splitlines():
            s = line.strip()
            if not s:
                continue
            low = s.lower()
            if low.startswith("question:"):
                q = s.split(":", 1)[1].strip()
            elif low.startswith("answer:"):
                a = s.split(":", 1)[1].strip()
            elif low.startswith("reference:"):
                r = s.split(":", 1)[1].strip()
        return {"question": q, "answer": a, "reference": r}

    def generate(self, question: str, retrieved: List[Dict]) -> str:
        contexts = [r["document"] for r in retrieved]
        messages = build_prompt(question, contexts)
        try:
            answer = self.chat.generate(messages, system=SYSTEM_PROMPT)
            if answer:
                return answer
        except Exception:
            # fall back below
            pass

        # Fallback: use the top retrieved chunk's answer and append [1]
        if contexts:
            top = self._parse_chunk(contexts[0])
            base = top.get("answer") or ""
            return (base + (" [1]" if base else "")).strip() or "(không có kết quả)"
        return "(không có kết quả)"

    def answer(self, question: str, top_k: Optional[int] = None) -> Dict:
        hits = self.retrieve(question, top_k=top_k)
        answer = self.generate(question, hits)
        return {"answer": answer, "contexts": hits}
