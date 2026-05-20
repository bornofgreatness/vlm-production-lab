"""Verify package imports and metric logic without GPU."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.hallucination_probe import build_unanswerable_probes, score_hallucination_rate
from src.eval.metrics import aggregate_results, exact_match


def main() -> None:
    rows = [
        {
            "id": "1",
            "question": "What color is the sky?",
            "gold": "blue",
            "choices": ["blue", "green", "red"],
            "subject": "natural science",
            "prediction": "blue",
            "exact_match": True,
            "contains_match": True,
        },
        {
            "id": "2",
            "question": "What color is the sky?",
            "gold": "blue",
            "choices": ["blue", "green", "red"],
            "subject": "natural science",
            "prediction": "green",
            "exact_match": False,
            "contains_match": False,
        },
    ]
    assert exact_match("Blue!", "blue")
    metrics = aggregate_results(rows)
    assert metrics["accuracy_em"] == 0.5

    probes = build_unanswerable_probes(rows, n=1, seed=0)
    assert len(probes) == 1
    probe_preds = [{**probes[0], "prediction": "green"}]
    hall = score_hallucination_rate(probe_preds)
    assert hall["n"] == 1

    print("smoke OK:", metrics, hall)


if __name__ == "__main__":
    main()
