"""
LoRA supervised fine-tuning for Qwen2-VL on curated ScienceQA data.

Single-GPU friendly via 4-bit loading + gradient checkpointing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from src.data.curate import (
    _extract_scienceqa_row,
    balance_by_subject,
    filter_examples,
    train_val_split,
)


class ScienceQADataset(Dataset):
    """Lazy ScienceQA examples for Trainer."""

    def __init__(self, hf_dataset: Any, processor: Any, max_samples: int | None = None):
        self.ds = hf_dataset.select(range(min(len(hf_dataset), max_samples or len(hf_dataset))))
        self.processor = processor
        self.images = self.ds["image"] if "image" in self.ds.column_names else None

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.ds[idx]
        image = self.images[idx] if self.images is not None else row["image"]
        question = row.get("question") or row.get("problem")
        choices = row["choices"]
        answer_idx = row["answer"]
        answer = choices[answer_idx] if isinstance(answer_idx, int) else str(answer_idx)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": str(question)},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(answer)}],
            },
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs


class VLMDataCollator:
    """Pad batch for variable-length multimodal inputs."""

    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        keys = features[0].keys()
        batch: dict[str, Any] = {}
        for key in keys:
            vals = [f[key] for f in features]
            if key == "labels":
                batch[key] = torch.nn.utils.rnn.pad_sequence(vals, batch_first=True, padding_value=-100)
            elif isinstance(vals[0], torch.Tensor):
                batch[key] = torch.nn.utils.rnn.pad_sequence(vals, batch_first=True, padding_value=0)
            else:
                batch[key] = vals
        return batch


def build_train_dataset(cfg: dict[str, Any], processor: Any) -> tuple[ScienceQADataset, ScienceQADataset | None]:
    """Build train/val HF subsets after in-memory curation (filter + balance + split)."""
    seed = cfg.get("seed", 42)
    max_n = cfg.get("max_train_samples") or 512
    ds = load_dataset(cfg["dataset_name"], cfg["dataset_config"], split="train")
    ds = ds.shuffle(seed=seed).select(range(min(len(ds), max_n * 2)))

    images = ds["image"] if "image" in ds.column_names else None
    examples = []
    for i in range(len(ds)):
        ex = _extract_scienceqa_row(ds[i], images, i)
        if ex:
            examples.append(ex)

    examples = filter_examples(examples)
    max_per = max(8, max_n // 8)
    examples = balance_by_subject(examples, max_per, seed)
    train_ex, val_ex = train_val_split(examples, cfg.get("val_ratio", 0.1), seed)

    train_hf = ds.select(range(min(len(ds), len(train_ex))))
    val_hf = (
        ds.select(range(len(train_ex), min(len(ds), len(train_ex) + len(val_ex))))
        if val_ex
        else None
    )
    return ScienceQADataset(train_hf, processor), (
        ScienceQADataset(val_hf, processor) if val_hf and len(val_hf) > 0 else None
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_lora.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    model_id = cfg["model_id"]
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if cfg.get("load_in_4bit") and torch.cuda.is_available():
        load_kwargs["load_in_4bit"] = True

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    if cfg.get("load_in_4bit"):
        model = prepare_model_for_kbit_training(model)
    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    train_ds, eval_ds = build_train_dataset(cfg, processor)
    collator = VLMDataCollator(processor)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=cfg.get("eval_steps", 100) if eval_ds else None,
        bf16=cfg.get("bf16", False) and torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    train_result = trainer.train()
    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))

    metrics = {"train_loss": train_result.training_loss, "output_dir": str(out_dir)}
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
