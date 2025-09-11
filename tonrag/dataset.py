from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import os
import pandas as pd
from datasets import load_dataset


def load_hf_dataset(name: str, split: str):
    return load_dataset(name, split=split)


def suggest_fields(columns: List[str]) -> Dict[str, List[str]]:
    cols_lower = [c.lower() for c in columns]
    guesses = {
        "text": [c for c in columns if c.lower() in ("text", "context", "passage", "document", "content", "reference")],
        "id": [c for c in columns if c.lower() in ("id", "doc_id", "uid", "no")],
        "question": [c for c in columns if c.lower() in ("question", "query", "q", "instruction")],
        "answer": [c for c in columns if c.lower() in ("answer", "answers", "a", "output")],
    }
    return guesses


def ensure_field(name: str, columns: List[str], fallback_list: List[str]) -> str:
    for candidate in fallback_list:
        if candidate in columns:
            return candidate
    # Try case-insensitive match
    lower_map = {c.lower(): c for c in columns}
    for candidate in fallback_list:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError(f"Could not find required field '{name}'. Available columns: {columns}")


def get_fields(
    dataset,
    text_field: Optional[str] = None,
    id_field: Optional[str] = None,
    question_field: Optional[str] = None,
    answer_field: Optional[str] = None,
    require_text: bool = True,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    cols = list(dataset.features.keys())

    # Resolve text field
    if text_field is None:
        if require_text:
            text_field = ensure_field(
                "text",
                cols,
                [
                    "context",
                    "text",
                    "document",
                    "content",
                    "passage",
                    "body",
                    "paragraph",
                    "article",
                    "reference",
                ],
            )
        else:
            text_field = None
    else:
        # Validate or case-insensitive match
        if text_field not in dataset.features:
            lower_map = {c.lower(): c for c in cols}
            if text_field.lower() in lower_map:
                text_field = lower_map[text_field.lower()]
            else:
                if require_text:
                    guesses = suggest_fields(cols).get("text", [])
                    hint = f" Try one of: {guesses}" if guesses else ""
                    raise ValueError(
                        f"Text field '{text_field}' not found in dataset. Available: {cols}.{hint}"
                    )
                else:
                    text_field = None

    # Optional fields: tolerate absence, but case-normalize if present
    if id_field is not None and id_field not in dataset.features:
        lower_map = {c.lower(): c for c in cols}
        id_field = lower_map.get(id_field.lower())
    if question_field is not None and question_field not in dataset.features:
        lower_map = {c.lower(): c for c in cols}
        question_field = lower_map.get(question_field.lower())
    if answer_field is not None and answer_field not in dataset.features:
        lower_map = {c.lower(): c for c in cols}
        answer_field = lower_map.get(answer_field.lower())

    return text_field, id_field, question_field, answer_field


class SimpleDataset:
    """Lightweight wrapper to make a pandas DataFrame look like a HF dataset
    for the limited interface we need: `.features`, `__len__`, and `__getitem__`.
    """

    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows
        columns = list(rows[0].keys()) if rows else []
        # mimic feature map behavior for membership checks and keys()
        self.features = {c: True for c in columns}

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._rows[idx]


def load_csv_dataset(path: str) -> SimpleDataset:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    # Convert NaNs to empty strings for robustness in chunking
    df = df.fillna("")
    rows: List[Dict[str, Any]] = df.to_dict(orient="records")
    return SimpleDataset(rows)
