import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    # Ollama is retained only for retrieval embeddings.
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:7860")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")

    # The only supported generation backend: GLM 5.3 served by vLLM.
    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://171.232.252.135:31585/v1")
    vllm_api_key: str = os.getenv("VLLM_API_KEY", "")
    vllm_model: str = os.getenv("VLLM_MODEL", "glm53-flash")

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
