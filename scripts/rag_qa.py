#!/usr/bin/env python3
import argparse
import os
import sys
import textwrap
from typing import List, Dict, Any, Tuple

import chromadb
from chromadb.utils import embedding_functions

# Allow direct execution with `python scripts/rag_qa.py`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tonrag.llm import get_default_chat


DB_PATH = os.path.join("data", "chroma_dmom")
COLLECTION_NAME = "dmom_qa"


def parse_chunk(doc: str) -> Dict[str, str]:
    # Expected format from build_vector_db.py
    # question: {input}\nanswer: {answer}\nreference: {ref}
    q, a, r = "", "", ""
    lines = [l.strip() for l in doc.splitlines() if l.strip()]
    for line in lines:
        if line.lower().startswith("question:"):
            q = line.split(":", 1)[1].strip()
        elif line.lower().startswith("answer:"):
            a = line.split(":", 1)[1].strip()
        elif line.lower().startswith("reference:"):
            r = line.split(":", 1)[1].strip()
    return {"question": q, "answer": a, "reference": r}


def retrieve(query: str, k: int) -> Dict[str, Any]:
    ef = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=DB_PATH)
    coll = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    return coll.query(query_texts=[query], n_results=k)


def build_context(res: Dict[str, Any], max_chars: int = 3000) -> Tuple[str, List[Dict[str, str]]]:
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    parsed = [parse_chunk(d or "") for d in docs]
    # Create numbered references (unique order-preserving)
    ref_order: List[str] = []
    for p in parsed:
        ref = p.get("reference", "")
        if ref and ref not in ref_order:
            ref_order.append(ref)

    # Map ref -> number
    ref_num = {ref: i + 1 for i, ref in enumerate(ref_order)}

    # Compose context lines prioritizing answers and references
    blocks: List[str] = []
    for idx, p in enumerate(parsed, start=1):
        ref = p.get("reference", "")
        num = ref_num.get(ref)
        cit = f"[{num}]" if num is not None else ""
        block = textwrap.dedent(
            f"""
            Nguồn {cit}:
            Hỏi: {p.get('question','')}
            Đáp: {p.get('answer','')}
            Tài liệu: {ref}
            """
        ).strip()
        blocks.append(block)

    context = "\n\n".join(blocks)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n..."

    # Return ordered list of refs for pretty printing
    ref_items = [{"n": i + 1, "ref": r} for i, r in enumerate(ref_order)]
    return context, ref_items


def generate_glm(prompt: str) -> str:
    return get_default_chat().generate(
        [{"role": "user", "content": prompt}],
        system="Bạn là bác sĩ trả lời câu hỏi y khoa một cách chính xác, súc tích, và có trích dẫn nguồn.",
    )


def main():
    parser = argparse.ArgumentParser(description="RAG over dmom data (Chroma + LLM)")
    parser.add_argument("query", type=str, help="User question (Vietnamese supported)")
    parser.add_argument("--k", type=int, default=5, help="Top-k documents to retrieve")
    parser.add_argument("--max-context", type=int, default=3000, help="Max characters in context")
    args = parser.parse_args()

    # Retrieve
    res = retrieve(args.query, args.k)
    context, refs = build_context(res, max_chars=args.max_context)

    # Build instruction in Vietnamese to match dataset
    prompt = textwrap.dedent(
        f"""
        Bạn là bác sĩ. Dựa trên các nguồn sau, hãy trả lời câu hỏi một cách chính xác, ngắn gọn, và trích dẫn số nguồn trong ngoặc vuông như [1], [2]. Nếu thông tin không đủ để trả lời, nói rõ: "Không đủ thông tin từ nguồn để trả lời đầy đủ".

        Ngữ cảnh:
        {context}

        Câu hỏi: {args.query}

        Đáp án (tiếng Việt, kèm trích dẫn [n]):
        """
    ).strip()

    answer = None
    try:
        answer = generate_glm(prompt)
    except Exception as e:
        print(f"[warn] GLM 5.3 generation failed: {e}")

    # Fallback: simple retrieval answer (top doc's answer)
    if not answer:
        docs = res.get("documents", [[]])[0]
        top = parse_chunk(docs[0]) if docs and docs[0] else {"answer": "(không có kết quả)", "reference": ""}
        # Attach first ref as [1] if exists
        if refs:
            answer = f"{top['answer']} [1]"
        else:
            answer = top["answer"]

    # Print
    print("Câu hỏi:")
    print(args.query)
    print("\nCâu trả lời:")
    print(answer)

    if refs:
        print("\nNguồn:")
        for item in refs:
            print(f"[{item['n']}] {item['ref']}")


if __name__ == "__main__":
    main()
