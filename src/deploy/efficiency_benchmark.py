"""
Efficiency benchmark: latency, VRAM, and optional 4-bit vs bf16 comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

from src.model_utils import generate_answer, load_qwen2_vl


def benchmark_inference(
    model_id: str,
    adapter_path: str | None,
    *,
    load_in_4bit: bool,
    n_warmup: int = 2,
    n_timed: int = 10,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "error": "CUDA not available",
            "note": "Run on a GPU machine or Colab for efficiency numbers",
        }

    torch.cuda.reset_peak_memory_stats()
    model, processor = load_qwen2_vl(model_id, adapter_path, load_in_4bit=load_in_4bit)

    ds = load_dataset("lmms-lab/ScienceQA", "ScienceQA-FULL", split="test")
    row = ds[0]
    image = ds[0]["image"]
    question = str(row.get("question") or row.get("problem"))

    for _ in range(n_warmup):
        generate_answer(model, processor, image, question, max_new_tokens=16)

    latencies: list[float] = []
    for _ in range(n_timed):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        generate_answer(model, processor, image, question, max_new_tokens=16)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    return {
        "load_in_4bit": load_in_4bit,
        "adapter": adapter_path,
        "latency_mean_ms": round(1000 * sum(latencies) / len(latencies), 2),
        "latency_p95_ms": round(1000 * sorted(latencies)[int(0.95 * len(latencies)) - 1], 2),
        "peak_vram_gb": round(vram_gb, 2),
        "n_timed": n_timed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--adapter", default="outputs/lora-sft")
    parser.add_argument("--out", default="results/efficiency.json")
    args = parser.parse_args()

    adapter = args.adapter if Path(args.adapter).exists() else None
    report = {
        "model_id": args.model_id,
        "configs": [],
    }
    for use_4bit in (True, False):
        try:
            report["configs"].append(
                benchmark_inference(args.model_id, adapter, load_in_4bit=use_4bit)
            )
        except Exception as exc:  # noqa: BLE001
            report["configs"].append({"load_in_4bit": use_4bit, "error": str(exc)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
