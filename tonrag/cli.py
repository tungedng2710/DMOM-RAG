from __future__ import annotations

import argparse
import os
from typing import List, Optional
from tqdm import tqdm

from .config import settings
from .dataset import load_hf_dataset, load_csv_dataset, suggest_fields, get_fields
from .embeddings import get_default_embeddings
from .vectorstore import ChromaStore
from .chunking import chunk_text
from .rag import RAGPipeline
try:
    from evaluation import rouge_l_corpus  # type: ignore
except Exception:
    rouge_l_corpus = None  # type: ignore


def _load_any_dataset(args: argparse.Namespace):
    if getattr(args, "csv", None):
        return load_csv_dataset(args.csv)
    if getattr(args, "dataset", None):
        return load_hf_dataset(args.dataset, split=args.split)
    raise ValueError("Provide either --csv <path> or --dataset <name>.")


def cmd_inspect(args: argparse.Namespace):
    ds = _load_any_dataset(args)
    cols = list(ds.features.keys())
    print(f"Dataset: {args.dataset} | split: {args.split}")
    print("Columns:", cols)
    print("Guesses:")
    print(suggest_fields(cols))
    print("First row:")
    print(ds[0])


def cmd_ingest(args: argparse.Namespace):
    ds = _load_any_dataset(args)
    try:
        text_field, id_field, _, _ = get_fields(
            ds, text_field=args.text_field, id_field=args.id_field, require_text=True
        )
    except ValueError as e:
        print(f"[ingest] {e}")
        print("Use 'python -m tonrag.cli inspect --dataset ... --split ...' to see columns.")
        return

    emb = get_default_embeddings()
    store = ChromaStore(persist_dir=settings.chroma_dir, collection_name=settings.collection_name, create_if_missing=True)

    ids: List[str] = []
    docs: List[str] = []
    metas: List[dict] = []

    # chunk and prepare
    for i in tqdm(range(len(ds)), desc="Chunking"):
        row = ds[i]
        base_id = str(row[id_field]) if id_field else str(i)
        text = row[text_field]
        chunks = chunk_text(text or "", chunk_size=args.chunk_size or settings.chunk_size, chunk_overlap=args.chunk_overlap or settings.chunk_overlap)
        for j, ch in enumerate(chunks):
            ids.append(f"{base_id}-{j}")
            docs.append(ch)
            metas.append({"row_id": base_id, "chunk": j})

    # embed in batches to avoid large payloads
    embeddings: List[List[float]] = []
    batch_size = args.batch_size
    for k in tqdm(range(0, len(docs), batch_size), desc="Embedding"):
        batch = docs[k:k+batch_size]
        embeddings.extend(emb.embed_documents(batch))

    # add to chroma
    store.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    print(f"Ingested {len(docs)} chunks into collection '{settings.collection_name}'.")


def _strip_markdown_html(s: str) -> str:
    if not s:
        return ""
    import re as _re
    # Remove code fences
    s = _re.sub(r"```[\s\S]*?```", "", s)
    # Inline code
    s = _re.sub(r"`([^`]+)`", r"\1", s)
    # Links [text](url) -> text (url)
    s = _re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r"\1 (\2)", s)
    # Bold/italic markers
    s = s.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
    # Headings markers
    s = _re.sub(r"^#{1,6}\s*", "", s, flags=_re.MULTILINE)
    # Strip HTML tags
    s = _re.sub(r"<[^>]+>", "", s)
    return s.strip()


def cmd_query(args: argparse.Namespace):
    rag = RAGPipeline(top_k=args.top_k, llm=getattr(args, 'llm', None))
    res = rag.answer(args.question, top_k=args.top_k)
    ans = res["answer"]
    if getattr(args, "strip_markdown", False):
        ans = _strip_markdown_html(ans)
    print("Answer:\n" + ans)
    print("\nRetrieved contexts:")
    for i, hit in enumerate(res["contexts"], 1):
        print(f"--- Context {i} (id={hit['id']}) ---")
        print(hit["document"])


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def cmd_eval(args: argparse.Namespace):
    ds = _load_any_dataset(args)
    _, _, q_field, a_field = get_fields(
        ds, text_field=args.text_field, question_field=args.question_field, answer_field=args.answer_field, require_text=False
    )
    if not q_field or not a_field:
        raise ValueError("Need --question-field and --answer-field (or auto-detected) for eval")

    rag = RAGPipeline(top_k=args.top_k, llm=getattr(args, 'llm', None))
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    total = 0
    correct = 0
    preds = []
    refs = []

    for i in tqdm(range(n), desc="Evaluating"):
        row = ds[i]
        q = row[q_field]
        gold = row[a_field]
        res = rag.answer(q, top_k=args.top_k)
        pred = res["answer"]
        # simple contains lexical matching
        if _normalize(gold) and _normalize(gold) in _normalize(pred):
            correct += 1
        total += 1
        preds.append(pred)
        refs.append(gold)

    acc = correct / max(total, 1)
    out = {"total": total, "correct": correct, "accuracy_contains": acc}
    if rouge_l_corpus is not None:
        try:
            rouge = rouge_l_corpus(preds, refs)
            out.update(rouge)
        except Exception:
            pass
    print(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tonrag")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="Inspect a dataset (HF or CSV) and suggest columns")
    pi.add_argument("--dataset", required=False)
    pi.add_argument("--csv", required=False, help="Path to CSV file")
    pi.add_argument("--split", default="train")
    pi.set_defaults(func=cmd_inspect)

    ping = sub.add_parser("ingest", help="Ingest dataset (HF or CSV) into Chroma")
    ping.add_argument("--dataset", required=False)
    ping.add_argument("--csv", required=False, help="Path to CSV file")
    ping.add_argument("--split", default="train")
    ping.add_argument("--text-field", default=None)
    ping.add_argument("--id-field", default=None)
    ping.add_argument("--chunk-size", type=int, default=None)
    ping.add_argument("--chunk-overlap", type=int, default=None)
    ping.add_argument("--batch-size", type=int, default=16)
    ping.set_defaults(func=cmd_ingest)

    pq = sub.add_parser("query", help="Ask a question against the indexed KB")
    pq.add_argument("--question", required=True)
    pq.add_argument("--top-k", type=int, default=settings.top_k)
    pq.add_argument("--llm", choices=["ollama", "gemini"], default=None, help="Choose chat backend (overrides CHAT_BACKEND)")
    pq.add_argument("--strip-markdown", action="store_true", help="Strip Markdown/HTML from answer for plain-text output")
    pq.set_defaults(func=cmd_query)

    pe = sub.add_parser("eval", help="Evaluate accuracy on dataset (HF or CSV)")
    pe.add_argument("--dataset", required=False)
    pe.add_argument("--csv", required=False, help="Path to CSV file")
    pe.add_argument("--split", default="validation")
    pe.add_argument("--text-field", default=None)
    pe.add_argument("--question-field", default=None)
    pe.add_argument("--answer-field", default=None)
    pe.add_argument("--top-k", type=int, default=settings.top_k)
    pe.add_argument("--limit", type=int, default=None)
    pe.add_argument("--llm", choices=["ollama", "gemini"], default=None, help="Choose chat backend (overrides CHAT_BACKEND)")
    pe.set_defaults(func=cmd_eval)

    return p


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
