from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


Role = Literal["system", "user", "assistant"]


@dataclass
class TurnMessage:
    role: Role
    text: str | None = None
    image: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TurnMessage":
        return cls(
            role=data.get("role", "user"),
            text=data.get("text"),
            image=data.get("image"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role}
        if self.text is not None:
            result["text"] = self.text
        if self.image is not None:
            result["image"] = self.image
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class QcageSample:
    sample_id: str
    history: list[TurnMessage]
    query: TurnMessage
    target_image: str | None = None
    answer_text: str | None = None
    feature_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QcageSample":
        return cls(
            sample_id=str(data["sample_id"]),
            history=[TurnMessage.from_dict(item) for item in data.get("history", [])],
            query=TurnMessage.from_dict(data.get("query", {"role": "user"})),
            target_image=data.get("target_image"),
            answer_text=data.get("answer_text"),
            feature_path=data.get("feature_path"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sample_id": self.sample_id,
            "history": [item.to_dict() for item in self.history],
            "query": self.query.to_dict(),
        }
        if self.target_image is not None:
            result["target_image"] = self.target_image
        if self.answer_text is not None:
            result["answer_text"] = self.answer_text
        if self.feature_path is not None:
            result["feature_path"] = self.feature_path
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def resolve_paths(self, root: str | Path) -> "QcageSample":
        root = Path(root)

        def resolve(value: str | None) -> str | None:
            if value is None:
                return None
            path = Path(value)
            return str(path if path.is_absolute() else root / path)

        history = [
            TurnMessage(
                role=message.role,
                text=message.text,
                image=resolve(message.image),
                metadata=message.metadata,
            )
            for message in self.history
        ]
        query = TurnMessage(
            role=self.query.role,
            text=self.query.text,
            image=resolve(self.query.image),
            metadata=self.query.metadata,
        )
        return QcageSample(
            sample_id=self.sample_id,
            history=history,
            query=query,
            target_image=resolve(self.target_image),
            answer_text=self.answer_text,
            feature_path=resolve(self.feature_path),
            metadata=self.metadata,
        )


SOURCE_QUERY = "query"
SOURCE_HISTORY_IMAGE = "history_image"
SOURCE_ANSWER_TEXT = "answer_text"
DEFAULT_SOURCE_NAMES = [SOURCE_QUERY, SOURCE_HISTORY_IMAGE, SOURCE_ANSWER_TEXT]

