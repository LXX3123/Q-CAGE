from __future__ import annotations

from dataclasses import dataclass

from qcage.data.schema import QcageSample, TurnMessage


@dataclass
class SerializedContext:
    text: str
    image_paths: list[str]


def _format_message(message: TurnMessage) -> tuple[str, list[str]]:
    parts: list[str] = [f"<{message.role}>"]
    images: list[str] = []
    if message.text:
        parts.append(message.text.strip())
    if message.image:
        parts.append("<image>")
        images.append(message.image)
    return " ".join(parts), images


def serialize_for_vlm(sample: QcageSample, include_answer: bool = True) -> SerializedContext:
    """Serialize a sample in natural interaction order for the frozen VLM."""
    chunks: list[str] = []
    images: list[str] = []

    for message in sample.history:
        text, message_images = _format_message(message)
        chunks.append(text)
        images.extend(message_images)

    query_text, query_images = _format_message(sample.query)
    chunks.append(query_text)
    images.extend(query_images)

    if include_answer and sample.answer_text:
        chunks.append(f"<assistant> {sample.answer_text.strip()}")

    return SerializedContext(text="\n".join(chunks), image_paths=images)


def prompt_for_generator(sample: QcageSample) -> str:
    """Return the current-turn textual instruction for the generator text path."""
    return sample.query.text or ""

