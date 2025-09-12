from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# Ensure project root is importable when running `uvicorn app.main:app`
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tonrag.rag import RAGPipeline  # noqa: E402


class ChatRequest(BaseModel):
    message: str
    top_k: Optional[int] = 5

class DebugRetrieveRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class DebugAnswerRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


def make_app() -> FastAPI:
    app = FastAPI(title="RAG Chatbot", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files
    static_dir = os.path.join(APP_DIR, "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    rag = RAGPipeline()

    @app.get("/")
    def index():
        index_path = os.path.join(static_dir, "index.html")
        if not os.path.exists(index_path):
            raise HTTPException(status_code=404, detail="index.html not found")
        return FileResponse(index_path)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        q = (req.message or "").strip()
        if not q:
            raise HTTPException(status_code=400, detail="Missing 'message'")
        top_k = int(req.top_k or 5)
        try:
            result = rag.answer(q, top_k=top_k)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RAG error: {e}")
        contexts = [
            {
                "id": c.get("id"),
                "distance": c.get("distance"),
                "document": (c.get("document") or "")[:2000],
            }
            for c in (result.get("contexts") or [])
        ]
        return {"answer": result.get("answer", ""), "contexts": contexts}

    # Debug endpoints to bring dev checks into the app
    @app.get("/api/debug/config")
    def debug_config():
        from tonrag.config import settings as s
        # Note: Avoid scanning large collections; only return config here
        return {
            "CHROMA_DIR": s.chroma_dir,
            "CHROMA_COLLECTION": s.collection_name,
            "CHROMA_QUERY_MODE": getattr(s, "chroma_query_mode", "auto"),
            "OLLAMA_BASE_URL": s.ollama_base_url,
            "GENERATION_MODEL": s.generation_model,
            "EMBEDDING_MODEL": s.embedding_model,
            "TOP_K": s.top_k,
        }

    @app.get("/api/debug/collections")
    def debug_collections():
        import chromadb
        from chromadb.errors import NotFoundError
        from tonrag.config import settings as s
        client = chromadb.PersistentClient(path=s.chroma_dir)
        cols = []
        for c in client.list_collections():
            try:
                count = client.get_collection(c.name).count()
            except NotFoundError:
                count = 0
            cols.append({"name": c.name, "count": count})
        return {"collections": cols}

    @app.post("/api/debug/retrieve")
    def debug_retrieve(req: DebugRetrieveRequest):
        try:
            hits = rag.retrieve(req.query, top_k=int(req.top_k or 3))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"retrieve failed: {e}")
        ctx = [
            {
                "id": h.get("id"),
                "distance": h.get("distance"),
                "document": (h.get("document") or "")[:2000],
                "metadata": h.get("metadata") or {},
            }
            for h in hits
        ]
        return {"count": len(ctx), "contexts": ctx}

    @app.post("/api/debug/answer")
    def debug_answer(req: DebugAnswerRequest):
        try:
            res = rag.answer((req.question or "").strip(), top_k=int(req.top_k or 3))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"answer failed: {e}")
        contexts = [
            {
                "id": c.get("id"),
                "distance": c.get("distance"),
                "document": (c.get("document") or "")[:2000],
            }
            for c in (res.get("contexts") or [])
        ]
        return {"answer": res.get("answer", ""), "contexts": contexts}

    return app


app = make_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7865)
