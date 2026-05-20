"""Canonical multimodal training example schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class VLMExample:
    """One vision-language SFT example."""

    id: str
    image_path: str | None  # local path or None if image is PIL/bytes in memory
    question: str
    answer: str
    subject: str = "general"
    difficulty: str = "medium"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_messages(self) -> list[dict[str, Any]]:
        """Chat-style messages for Qwen2-VL style processors."""
        content: list[dict[str, Any]] = []
        if self.image_path:
            content.append({"type": "image", "path": self.image_path})
        content.append({"type": "text", "text": self.question})
        return [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": self.answer}]},
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
