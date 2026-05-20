"""
Multimodal evaluation harness: benchmarks + hallucination probe + report export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset
from tqdm import tqdm

from src.eval.hallucination_probe import build_unanswerable_probes, score_hallucination_rate
from src.eval.metrics import aggregate_results, contains_match, exact_match
from src.model_utils import generate_answer, load_qwen2_vl


def _scienceqa_rows(ds: Any, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    images = ds["image"] if "image" in ds.column_names else None
    n = min(len(ds), max_samples)
    for i in range(n):
        row = ds[i]
        choices = row.get("choices", [])
        ans_idx = row.get("answer")
        gold = choices[ans_idx] if isinstance(ans_idx, int) else str(ans_idx)
        rows.append(
            {
                "id": str(row.get("id", i)),
                "image": images[i] if images is not None else row.get("image"),
                "question": str(row.get("question") or row.get("problem")),
                "gold": str(gold),
                "choices": choices,
                "subject": str(row.get("subject", "general")),
            }
        )
    return rows


def _textvqa_rows(ds: Any, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = min(len(ds), max_samples)
    for i in range(n):
        row = ds[i]
        answers = row.get("answers") or row.get("answer")
        if isinstance(answers, list) and answers:
            gold = answers[0]
        else:
            gold = str(answers)
        rows.append(
            {
                "id": str(row.get("question_id", i)),
                "image": row["image"],
                "question": str(row.get("question")),
                "gold": str(gold),
                "subject": "textvqa",
            }
        )
    return rows


def run_benchmark(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="eval"):
        pred = generate_answer(
            model,
            processor,
            row["image"],
            row["question"],
            max_new_tokens=max_new_tokens,
        )
        results.append(
            {
                **row,
                "prediction": pred,
                "exact_match": exact_match(pred, row["gold"]),
                "contains_match": contains_match(pred, row["gold"]),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLM on benchmarks")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--adapter", default=None, help="Override adapter path")
    parser.add_argument("--smoke", action="store_true", help="Run on 4 samples only")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    adapter = args.adapter or cfg.get("adapter_path")
    if adapter and not Path(adapter).exists():
        adapter = None

    model, processor = load_qwen2_vl(
        cfg["model_id"],
        adapter,
        load_in_4bit=True,
    )

    max_tokens = cfg.get("max_new_tokens", 32)
    report: dict[str, Any] = {"model_id": cfg["model_id"], "adapter": adapter, "benchmarks": {}}

    for bench in cfg.get("benchmarks", []):
        name = bench["name"]
        max_samples = 4 if args.smoke else bench.get("max_samples", 64)
        if name == "scienceqa":
            ds = load_dataset(bench["dataset"], bench.get("config"), split=bench.get("split", "test"))
            rows = _scienceqa_rows(ds, max_samples)
        elif name == "textvqa":
            ds = load_dataset(bench["dataset"], split=bench.get("split", "validation"))
            rows = _textvqa_rows(ds, max_samples)
        else:
            continue

        results = run_benchmark(model, processor, rows, max_new_tokens=max_tokens)
        metrics = aggregate_results(results)
        report["benchmarks"][name] = metrics

        if name == "scienceqa":
            probes = build_unanswerable_probes(results, cfg.get("hallucination_probe_samples", 32))
            probe_preds = run_benchmark(model, processor, probes, max_new_tokens=max_tokens)
            for p, pred_row in zip(probes, probe_preds):
                p["prediction"] = pred_row["prediction"]
            report["hallucination_probe"] = score_hallucination_rate(probe_preds)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    tag = "finetuned" if adapter else "baseline"
    out_path = out_dir / f"eval_{tag}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
