from __future__ import annotations

import argparse
import json
import os
import sys
from http import HTTPStatus
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse


# Ensure project root is on sys.path so we can import tonrag.* when run from app/
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tonrag.rag import RAGPipeline  # noqa: E402


class RAGRequestHandler(SimpleHTTPRequestHandler):
    # Set static directory base
    static_dir = os.path.join(APP_DIR, "static")
    rag = RAGPipeline()

    def translate_path(self, path: str) -> str:
        # Serve files from static_dir
        if path == "/":
            path = "/index.html"
        # Resolve relative to static_dir
        rel = path.lstrip("/")
        return os.path.join(self.static_dir, rel)

    def end_headers(self):
        # Basic CORS for local usage if embedding in other hosts
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/chat":
            self._handle_chat()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def _json(self, obj, status=HTTPStatus.OK):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_chat(self):
        try:
            body = self._read_json()
            question = (body.get("message") or body.get("question") or "").strip()
            top_k = int(body.get("top_k") or 5)
            llm = (body.get("llm") or body.get("provider") or body.get("backend") or "").strip().lower() or None
            gemini_api_key = (body.get("gemini_api_key") or "").strip() or None
            if not question:
                return self._json({"error": "Missing 'message'"}, status=HTTPStatus.BAD_REQUEST)

            try:
                # Instantiate per request if a different LLM is requested
                if llm is None:
                    rag = self.rag
                else:
                    if llm == 'gemini':
                        rag = RAGPipeline(top_k=top_k, llm=llm, gemini_api_key=gemini_api_key)
                    else:
                        rag = RAGPipeline(top_k=top_k, llm=llm)
                result = rag.answer(question, top_k=top_k)
            except Exception as e:
                return self._json({"error": f"RAG error: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

            # Reduce context size for payload
            contexts = [
                {
                    "id": c.get("id"),
                    "distance": c.get("distance"),
                    "document": c.get("document", "")[:2000],
                }
                for c in (result.get("contexts") or [])
            ]
            return self._json({
                "answer": result.get("answer", ""),
                "contexts": contexts,
            })
        except Exception as e:
            return self._json({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


def main():
    parser = argparse.ArgumentParser(description="Simple RAG chatbot web app")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7865)
    args = parser.parse_args()

    os.makedirs(os.path.join(APP_DIR, "static"), exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), RAGRequestHandler)
    print(f"Serving on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
