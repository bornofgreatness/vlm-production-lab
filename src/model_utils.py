"""Shared model loading for Qwen2-VL + optional LoRA adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def load_qwen2_vl(
    model_id: str,
    adapter_path: str | None = None,
    *,
    load_in_4bit: bool = False,
    device_map: str | dict | None = "auto",
) -> tuple[Any, Any]:
    """Load base VLM and optional LoRA weights."""
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": device_map,
    }
    if load_in_4bit:
        kwargs["load_in_4bit"] = True

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, trust_remote_code=True, **kwargs)

    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

    return model, processor


def generate_answer(
    model: Any,
    processor: Any,
    image: Any,
    question: str,
    *,
    max_new_tokens: int = 32,
) -> str:
    """Single VQA inference."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Decode only new tokens
    input_len = inputs["input_ids"].shape[1]
    generated = out[:, input_len:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
