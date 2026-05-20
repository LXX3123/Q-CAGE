from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    for key in ["data", "samples", "trajectories", "items"]:
        if isinstance(data.get(key), list):
            return data[key]
    raise ValueError(f"Could not find a record list in {path}")


def _message(role: str, text: str | None = None, image: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"role": role}
    if text:
        result["text"] = text
    if image:
        result["image"] = image
    return result


def _as_messages(value) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        messages = []
        for item in value:
            if isinstance(item, dict):
                messages.append(
                    _message(
                        item.get("role", "user"),
                        item.get("text") or item.get("content") or item.get("prompt"),
                        item.get("image") or item.get("image_path") or item.get("img"),
                    )
                )
            else:
                messages.append(_message("user", str(item)))
        return messages
    if isinstance(value, str):
        return [_message("user", value)]
    return []


def convert_record(record: dict[str, Any], image_root: str | None = None) -> dict[str, Any]:
    sample_id = str(
        record.get("sample_id")
        or record.get("id")
        or record.get("uid")
        or record.get("trajectory_id")
        or record.get("index")
    )
    history = _as_messages(record.get("history") or record.get("context") or record.get("dialogue"))

    query_value = record.get("query") or record.get("current_turn") or record.get("question")
    if isinstance(query_value, dict):
        query = _message(
            query_value.get("role", "user"),
            query_value.get("text") or query_value.get("content") or query_value.get("prompt"),
            query_value.get("image") or query_value.get("image_path") or query_value.get("img"),
        )
    else:
        query = _message(
            "user",
            str(query_value or record.get("prompt") or record.get("instruction") or ""),
            record.get("query_image") or record.get("current_image") or record.get("image"),
        )

    return {
        "sample_id": sample_id,
        "history": history,
        "query": query,
        "target_image": record.get("target_image") or record.get("answer_image") or record.get("gt_image"),
        "answer_text": record.get("answer_text") or record.get("vlm_answer") or record.get("rationale"),
        "metadata": {
            "source": "opening_adapter",
            "image_root": image_root,
            "raw_keys": sorted(record.keys()),
        },
    }


def convert_file(input_path: str | Path, output_jsonl: str | Path, image_root: str | None = None) -> Path:
    records = _read_records(input_path)
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(convert_record(record, image_root=image_root)) + "\n")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an OpenING-style JSON/JSONL file to Q-CAGE JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--image-root", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(convert_file(args.input, args.output_jsonl, image_root=args.image_root))


if __name__ == "__main__":
    main()
