from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    text = text or ""
    if chunk_size <= 0:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks

