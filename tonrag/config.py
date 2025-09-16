import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    # Ollama / API
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:7860")
    generation_model: str = os.getenv("GENERATION_MODEL", "gpt-oss:20b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")

    # Gemini / API
    # Default to public Google Generative Language API v1 endpoint
    gemini_base_url: str = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # Default chat backend: 'ollama' or 'gemini'
    chat_backend: str = os.getenv("CHAT_BACKEND", "ollama")

    # Vector store
    # Default to the DB built by scripts/build_vector_db.py
    chroma_dir: str = os.getenv("CHROMA_DIR", os.path.abspath("./data/chroma_dmom"))
    collection_name: str = os.getenv("CHROMA_COLLECTION", "dmom_qa")
    # Query mode: 'text' uses Chroma's embedding function (if configured),
    # 'embed' uses our embedding client, 'auto' tries text then falls back to embed.
    chroma_query_mode: str = os.getenv("CHROMA_QUERY_MODE", "auto")

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "5"))


settings = Settings()
