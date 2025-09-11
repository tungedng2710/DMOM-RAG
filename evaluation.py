from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


def _normalize_to_words(text: str) -> List[str]:
    if text is None:
        return []
    # Lowercase and keep only alphanumerics and spaces, then split
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


def _lcs_len(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    # DP over two rows to save memory
    prev = [0] * (m + 1)
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev, cur = cur, prev
    return prev[m]


@dataclass
class RougeL:
    precision: float
    recall: float
    f1: float
    lcs: int
    pred_len: int
    ref_len: int


def rouge_l_score(pred: str, ref: str, beta: float = 1.2) -> RougeL:
    pred_toks = _normalize_to_words(pred)
    ref_toks = _normalize_to_words(ref)

    lcs = _lcs_len(pred_toks, ref_toks)
    pred_len = len(pred_toks)
    ref_len = len(ref_toks)
    if pred_len == 0 or ref_len == 0 or lcs == 0:
        return RougeL(precision=0.0, recall=0.0, f1=0.0, lcs=lcs, pred_len=pred_len, ref_len=ref_len)

    p = lcs / pred_len
    r = lcs / ref_len
    beta2 = beta * beta
    f1 = (1 + beta2) * p * r / (r + beta2 * p) if (r + beta2 * p) > 0 else 0.0
    return RougeL(precision=p, recall=r, f1=f1, lcs=lcs, pred_len=pred_len, ref_len=ref_len)


def rouge_l_corpus(preds: List[str], refs: List[str], beta: float = 1.2) -> Dict[str, float]:
    assert len(preds) == len(refs), "preds and refs length mismatch"
    total_lcs = 0
    total_pred_len = 0
    total_ref_len = 0
    f1s: List[float] = []

    for pred, ref in zip(preds, refs):
        s = rouge_l_score(pred, ref, beta=beta)
        total_lcs += s.lcs
        total_pred_len += max(s.pred_len, 1)
        total_ref_len += max(s.ref_len, 1)
        f1s.append(s.f1)

    # Micro-averaged P/R over the corpus
    micro_p = total_lcs / total_pred_len if total_pred_len > 0 else 0.0
    micro_r = total_lcs / total_ref_len if total_ref_len > 0 else 0.0
    beta2 = beta * beta
    micro_f1 = (1 + beta2) * micro_p * micro_r / (micro_r + beta2 * micro_p) if (micro_r + beta2 * micro_p) > 0 else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "rougeL_precision_micro": micro_p,
        "rougeL_recall_micro": micro_r,
        "rougeL_f1_micro": micro_f1,
        "rougeL_f1_macro": macro_f1,
        "count": float(len(preds)),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute ROUGE-L between two text columns in a CSV")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--pred-col", required=True, help="Predictions column name")
    parser.add_argument("--ref-col", required=True, help="References column name")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of rows")
    parser.add_argument("--beta", type=float, default=1.2, help="Beta for F-score (default 1.2)")
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.csv).fillna("")
    if args.limit is not None:
        df = df.head(args.limit)
    preds = df[args.pred_col].astype(str).tolist()
    refs = df[args.ref_col].astype(str).tolist()
    out = rouge_l_corpus(preds, refs, beta=args.beta)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
