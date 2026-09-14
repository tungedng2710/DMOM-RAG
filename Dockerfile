# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install only the dependencies required by the API and web UI. Dataset
# ingestion and evaluation dependencies remain in requirements.txt for local use.
COPY requirements.docker.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.docker.txt

# Using OllamaEmbeddings (HTTP to OLLAMA_BASE_URL) per app defaults
# No sentence-transformers model prefetch is needed here.

# Copy application code. Runtime data is mounted by Docker Compose.
COPY . .

# Expose FastAPI port
EXPOSE 7865

# Default environment (can be overridden by docker-compose env_file)
ENV HOST=0.0.0.0 PORT=7865

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7865/health', timeout=3)" || exit 1

# Start the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7865"]
