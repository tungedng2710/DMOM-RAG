from __future__ import annotations

from typing import List, Dict, Any
import os
import chromadb

from .config import settings


class ChromaStore:
    def __init__(self, collection_name: str | None = None, persist_dir: str | None = None):
        self.persist_dir = persist_dir or settings.chroma_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        # Use PersistentClient to be compatible with on-disk DBs created elsewhere
        # and avoid telemetry issues.
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name or settings.collection_name)

    def add(self, ids: List[str], documents: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]] | None = None):
        self.collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def _pack(self, res: Dict[str, Any]):
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out = []
        for i in range(len(docs)):
            out.append({
                "id": ids[i],
                "document": docs[i],
                "metadata": metas[i] if metas and i < len(metas) else {},
                "distance": dists[i] if dists and i < len(dists) else None,
            })
        return out

    def query(self, query_embedding: List[float], top_k: int = 5):
        res = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return self._pack(res)

    def query_text(self, query_text: str, top_k: int = 5):
        res = self.collection.query(query_texts=[query_text], n_results=top_k)
        return self._pack(res)
