#!/usr/bin/env python3
import csv
import os
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions


CSV_PATH = os.path.join("data", "dmom_data.csv")
DB_PATH = os.path.join("data", "chroma_dmom")
COLLECTION_NAME = "dmom_qa"


def read_rows(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_chunk(row: Dict[str, str]) -> str:
    q = (row.get("input") or "").strip()
    out = (row.get("output") or "").strip()
    manual = (row.get("Manually review") or "").strip()
    ref = (row.get("Reference") or "").strip()

    answer = manual if manual else out

    return f"question: {q}\nanswer: {answer}\nreference: {ref}"


def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"CSV not found: {CSV_PATH}")

    os.makedirs(DB_PATH, exist_ok=True)

    # Embedding function (local, no API key required)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=DB_PATH)

    # Re-create collection to ensure a clean ingest
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    coll = client.create_collection(name=COLLECTION_NAME, embedding_function=ef)

    rows = read_rows(CSV_PATH)
    print(f"Loaded {len(rows)} rows from {CSV_PATH}")

    documents: List[str] = []
    metadatas: List[Dict[str, str]] = []
    ids: List[str] = []

    # Prepare a small preview file for manual inspection
    preview_path = os.path.join("data", "dmom_chunks_preview.txt")
    with open(preview_path, "w", encoding="utf-8") as preview:
        preview.write("Preview of formatted chunks (first 25):\n\n")

        for idx, row in enumerate(rows):
            q = (row.get("input") or "").strip()
            if not q:
                # Skip rows without a question
                continue

            chunk = build_chunk(row)
            if idx < 25:
                preview.write(chunk + "\n---\n")

            doc_id = str(row.get("no") or idx)
            if not doc_id:
                doc_id = str(idx)

            documents.append(chunk)
            metadatas.append(
                {
                    "no": doc_id,
                    "reference": (row.get("Reference") or "").strip(),
                    "source": "dmom_data.csv",
                    "has_manual_review": "true" if (row.get("Manually review") or "").strip() else "false",
                    "instruction": (row.get("instruction") or "").strip(),
                }
            )
            ids.append(f"dmom-{doc_id}")

    print(f"Prepared {len(documents)} chunks for embedding")

    # Ingest in batches to manage memory/throughput
    batch_size = 256
    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        coll.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        print(f"Ingested {min(end, len(documents))}/{len(documents)}")

    # Basic verification
    count = coll.count()
    print(f"Collection '{COLLECTION_NAME}' contains {count} documents")
    print(f"Chroma DB path: {DB_PATH}")
    print(f"Preview saved to: {preview_path}")


if __name__ == "__main__":
    main()

