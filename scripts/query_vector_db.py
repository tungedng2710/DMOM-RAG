#!/usr/bin/env python3
import argparse
import os

import chromadb
from chromadb.utils import embedding_functions


DB_PATH = os.path.join("data", "chroma_dmom")
COLLECTION_NAME = "dmom_qa"


def main():
    parser = argparse.ArgumentParser(description="Query the local Chroma vector DB")
    parser.add_argument("query", type=str, help="Search query text")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=DB_PATH)
    coll = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

    res = coll.query(query_texts=[args.query], n_results=args.k)

    docs = res.get("documents", [[]])[0]
    ids = res.get("ids", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances") or res.get("embeddings") or [[]]
    dists = dists[0] if dists else [None] * len(docs)

    for i, (doc, _id, meta, dist) in enumerate(zip(docs, ids, metas, dists), start=1):
        print(f"#{i}")
        print(f"id: {_id}")
        if dist is not None:
            print(f"distance: {dist}")
        print(f"meta: {meta}")
        print("---")
        print(doc)
        print("\n====\n")


if __name__ == "__main__":
    main()

