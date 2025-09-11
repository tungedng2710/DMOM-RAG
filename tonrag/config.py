import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    # Ollama / API
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:7860")
    generation_model: str = os.getenv("GENERATION_MODEL", "gpt-oss:20b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    # Vector store
    chroma_dir: str = os.getenv("CHROMA_DIR", os.path.abspath("./data/chroma"))
    collection_name: str = os.getenv("CHROMA_COLLECTION", "dmom_collection")

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "5"))


settings = Settings()

