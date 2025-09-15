# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (git for some pip deps; build tools for wheels when needed)
RUN apt-get update \ 
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Optional: pre-download sentence-transformers model to speed up first run
RUN python - <<'PY'
from chromadb.utils import embedding_functions
try:
    embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    print("[docker] Pre-fetched all-MiniLM-L6-v2 model")
except Exception as e:
    print("[docker] Skipping model prefetch:", e)
PY

# Copy application code and data (including prebuilt Chroma DB)
COPY . .

# Expose FastAPI port
EXPOSE 7865

# Default environment (can be overridden by docker-compose env_file)
ENV HOST=0.0.0.0 PORT=7865

# Start the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7865"]
