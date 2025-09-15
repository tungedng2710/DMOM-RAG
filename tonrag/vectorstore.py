from __future__ import annotations

from typing import List, Dict, Any
import os
import chromadb

from .config import settings


class ChromaStore:
    def __init__(self, collection_name: str | None = None, persist_dir: str | None = None, create_if_missing: bool = False):
        # Resolve persist dir; if relative, anchor to project root (dir of this file/..)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        cfg_dir = persist_dir or settings.chroma_dir
        if not os.path.isabs(cfg_dir):
            cfg_dir = os.path.abspath(os.path.join(base_dir, cfg_dir))
        self.persist_dir = cfg_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        # Use PersistentClient to be compatible with on-disk DBs created elsewhere
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        name = collection_name or settings.collection_name

        # For text queries without external embedding servers, prefer to attach
        # the same SentenceTransformerEmbeddingFunction used during ingest
        embed_fn = None
        try:
            from chromadb.utils import embedding_functions as _ef
            embed_fn = _ef.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        except Exception:
            embed_fn = None  # optional; query_text will fallback in RAGPipeline

        if create_if_missing:
            self.collection = self.client.get_or_create_collection(name=name, embedding_function=embed_fn)
        else:
            # Raise error if not found to avoid silently querying an empty collection
            try:
                if embed_fn is not None:
                    self.collection = self.client.get_collection(name, embedding_function=embed_fn)
                else:
                    self.collection = self.client.get_collection(name)
            except Exception as e:
                raise RuntimeError(
                    f"Chroma collection '{name}' not found in '{self.persist_dir}'. Verify CHROMA_DIR/CHROMA_COLLECTION or build the DB."
                ) from e

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
