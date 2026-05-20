"""Evaluation metrics for VQA / VLM benchmarks."""

from __future__ import annotations

import re
import string
from collections import defaultdict
from typing import Any


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    # remove articles / punctuation for VQA-style matching
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def contains_match(pred: str, gold: str) -> bool:
    p, g = normalize_answer(pred), normalize_answer(gold)
    return g in p or p in g


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute accuracy and per-subject breakdown."""
    total = len(rows)
    if total == 0:
        return {"accuracy_em": 0.0, "accuracy_contains": 0.0, "n": 0}

    em = sum(1 for r in rows if r.get("exact_match")) / total
    cm = sum(1 for r in rows if r.get("contains_match")) / total
    by_subject: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_subject[r.get("subject", "general")].append(bool(r.get("exact_match")))
    subject_acc = {k: sum(v) / len(v) for k, v in by_subject.items()}

    return {
        "n": total,
        "accuracy_em": round(em, 4),
        "accuracy_contains": round(cm, 4),
        "per_subject_em": {k: round(v, 4) for k, v in subject_acc.items()},
    }
