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

    return app


app = make_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7865)

