"""
Multimodal dataset curation: filtering, balancing, and train/val splits.

Designed for ScienceQA-style VQA but works with any HF dataset that exposes
image + question + answer fields.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

from src.data.schema import VLMExample


def _extract_scienceqa_row(row: dict[str, Any], images: Any, idx: int) -> VLMExample | None:
    """Map a ScienceQA row to VLMExample."""
    try:
        image = images[idx] if images is not None else row.get("image")
        if image is None:
            return None
        question = row.get("question") or row.get("problem")
        choices = row.get("choices")
        answer_idx = row.get("answer")
        if question is None or choices is None or answer_idx is None:
            return None
        if isinstance(answer_idx, int):
            answer = choices[answer_idx]
        else:
            answer = str(answer_idx)
        subject = str(row.get("subject", "general"))
        grade = row.get("grade", "unknown")
        return VLMExample(
            id=str(row.get("id", idx)),
            image_path=None,  # PIL image stored separately at export
            question=str(question).strip(),
            answer=str(answer).strip(),
            subject=subject,
            difficulty=str(grade),
            metadata={"choices": choices, "lecture": row.get("lecture"), "solution": row.get("solution")},
        )
    except (IndexError, KeyError, TypeError):
        return None


def filter_examples(
    examples: list[VLMExample],
    *,
    min_answer_len: int = 1,
    max_answer_len: int = 128,
    drop_empty_questions: bool = True,
) -> list[VLMExample]:
    """Quality filters for robust SFT data."""
    kept: list[VLMExample] = []
    for ex in examples:
        if drop_empty_questions and not ex.question.strip():
            continue
        alen = len(ex.answer.strip())
        if alen < min_answer_len or alen > max_answer_len:
            continue
        kept.append(ex)
    return kept


def balance_by_subject(examples: list[VLMExample], max_per_subject: int, seed: int = 42) -> list[VLMExample]:
    """Cap samples per subject to reduce dominance of large categories."""
    rng = random.Random(seed)
    by_subject: dict[str, list[VLMExample]] = {}
    for ex in examples:
        by_subject.setdefault(ex.subject, []).append(ex)
    balanced: list[VLMExample] = []
    for subject, items in by_subject.items():
        rng.shuffle(items)
        balanced.extend(items[:max_per_subject])
    rng.shuffle(balanced)
    return balanced


def train_val_split(examples: list[VLMExample], val_ratio: float, seed: int) -> tuple[list[VLMExample], list[VLMExample]]:
    rng = random.Random(seed)
    shuffled = examples.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def load_scienceqa(
    dataset_name: str,
    config: str,
    split: str,
    max_samples: int | None,
    seed: int,
) -> tuple[list[VLMExample], list[Any]]:
    ds = load_dataset(dataset_name, config, split=split)
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=seed).select(range(max_samples))
    images = ds["image"] if "image" in ds.column_names else None
    examples: list[VLMExample] = []
    pil_images: list[Any] = []
    for i, row in enumerate(ds):
        ex = _extract_scienceqa_row(row, images, i)
        if ex is None:
            continue
        examples.append(ex)
        if images is not None:
            pil_images.append(images[i])
    return examples, pil_images


def export_jsonl(
    examples: list[VLMExample],
    image_by_id: dict[str, Any],
    out_dir: Path,
    prefix: str,
) -> Path:
    """Export curated examples; images saved as PNG alongside jsonl."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / f"{prefix}_images"
    img_dir.mkdir(exist_ok=True)
    jsonl_path = out_dir / f"{prefix}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            record = ex.to_dict()
            img = image_by_id.get(ex.id)
            if img is not None:
                img_path = img_dir / f"{ex.id}.png"
                img.save(img_path)
                record["image_path"] = str(img_path)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return jsonl_path


def curation_report(examples: list[VLMExample]) -> dict[str, Any]:
    subjects = Counter(ex.subject for ex in examples)
    difficulties = Counter(ex.difficulty for ex in examples)
    return {
        "total": len(examples),
        "subjects": dict(subjects),
        "difficulties": dict(difficulties),
        "avg_question_len": sum(len(e.question) for e in examples) / max(len(examples), 1),
        "avg_answer_len": sum(len(e.answer) for e in examples) / max(len(examples), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate multimodal SFT dataset")
    parser.add_argument("--dataset", default="lmms-lab/ScienceQA")
    parser.add_argument("--config", default="ScienceQA-FULL")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--max-per-subject", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/curated")
    args = parser.parse_args()

    examples, pil_images = load_scienceqa(
        args.dataset, args.config, args.split, args.max_samples, args.seed
    )
    before = len(examples)
    image_by_id = {ex.id: pil_images[i] for i, ex in enumerate(examples) if i < len(pil_images)}
    examples = filter_examples(examples)
    examples = balance_by_subject(examples, args.max_per_subject, args.seed)
    train_ex, val_ex = train_val_split(examples, args.val_ratio, args.seed)

    out = Path(args.out_dir)
    train_paths = export_jsonl(train_ex, image_by_id, out, "train")
    val_paths = export_jsonl(val_ex, image_by_id, out, "val")

    report = {
        "source": {"dataset": args.dataset, "config": args.config, "split": args.split},
        "counts": {"raw": before, "after_filter_balance": len(examples), "train": len(train_ex), "val": len(val_ex)},
        "train": curation_report(train_ex),
        "val": curation_report(val_ex),
        "outputs": {"train": str(train_paths), "val": str(val_paths)},
    }
    report_path = out / "curation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
