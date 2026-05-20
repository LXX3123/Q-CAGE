from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def load_opening_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def opening_input_to_qcage(record: dict[str, Any], *, split_outputs: bool = False) -> list[dict[str, Any]]:
    """Convert official OpenING records into Q-CAGE JSONL records.

    OpenING stores each instance as `conversations[0].input` and
    `conversations[1].output`. For benchmark inference we create one Q-CAGE
    sample per OpenING instance. For training, `split_outputs=True` creates one
    sample per target output image.
    """
    inputs = record.get("conversations", [{}])[0].get("input", [])
    outputs = record.get("conversations", [{}, {}])[1].get("output", [])
    total_uid = str(record["total_uid"])
    base_metadata = {
        "opening": {
            "total_uid": total_uid,
            "meta_task_id": record.get("meta_task_id"),
            "meta_task_name": record.get("meta_task_name"),
            "subtask_id": record.get("subtask_id"),
            "subtask_name": record.get("subtask_name"),
            "data_id": record.get("data_id"),
            "input": inputs,
            "target_output": outputs,
        }
    }

    history = [
        {
            "role": "user",
            "text": item.get("text"),
            "image": item.get("image"),
        }
        for item in inputs[:-1]
    ]
    last_input = inputs[-1] if inputs else {"text": "", "image": None}
    query = {"role": "user", "text": last_input.get("text"), "image": last_input.get("image")}

    if not split_outputs:
        first_output = outputs[0] if outputs else {}
        return [
            {
                "sample_id": total_uid,
                "history": history,
                "query": query,
                "target_image": first_output.get("image"),
                "answer_text": first_output.get("text"),
                "metadata": base_metadata,
            }
        ]

    samples: list[dict[str, Any]] = []
    rolling_history = history + [query]
    for index, output in enumerate(outputs):
        sample_metadata = json.loads(json.dumps(base_metadata))
        sample_metadata["opening"]["output_index"] = index
        samples.append(
            {
                "sample_id": f"{total_uid}_out{index}",
                "history": rolling_history,
                "query": {
                    "role": "user",
                    "text": output.get("text") or "Generate the next image for this interleaved output.",
                    "image": None,
                },
                "target_image": output.get("image"),
                "answer_text": output.get("text"),
                "metadata": sample_metadata,
            }
        )
        rolling_history = rolling_history + [
            {"role": "assistant", "text": output.get("text"), "image": output.get("image")}
        ]
    return samples


def convert_opening_file(
    *,
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    split_outputs: bool = False,
) -> Path:
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in load_opening_jsonl(input_jsonl):
            for sample in opening_input_to_qcage(record, split_outputs=split_outputs):
                handle.write(json.dumps(sample) + "\n")
    return output_path


def _opening_metadata(sample) -> dict[str, Any]:
    opening = (getattr(sample, "metadata", {}) or {}).get("opening")
    if opening is None:
        return {
            "total_uid": sample.sample_id,
            "meta_task_id": None,
            "meta_task_name": None,
            "subtask_id": None,
            "subtask_name": None,
            "data_id": sample.sample_id,
            "input": [],
            "target_output": [],
        }
    return opening


def write_official_output(
    *,
    sample,
    generation: dict[str, Any],
    model_output_dir: str | Path,
    answer_text: str | None = None,
) -> Path:
    """Write one sample in official OpenING `gen_outputs/MODEL_output` format."""
    model_output_dir = Path(model_output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    opening = _opening_metadata(sample)
    total_uid = str(opening.get("total_uid") or sample.sample_id).split("_out")[0]
    images = generation.get("images") or []
    output_items: list[dict[str, Any]] = []

    if images:
        for index, image_path in enumerate(images):
            suffix = Path(image_path).suffix or ".jpg"
            out_name = f"{total_uid}-o-{index}{suffix}"
            out_path = model_output_dir / out_name
            if Path(image_path).resolve() != out_path.resolve():
                shutil.copyfile(image_path, out_path)
            text = answer_text
            targets = opening.get("target_output") or []
            if text is None and index < len(targets):
                text = targets[index].get("text")
            output_items.append({"text": text or "", "image": out_name})
    else:
        output_items.append({"text": answer_text or "", "image": None})

    record = {
        "meta_task_id": opening.get("meta_task_id"),
        "meta_task_name": opening.get("meta_task_name"),
        "subtask_id": opening.get("subtask_id"),
        "subtask_name": opening.get("subtask_name"),
        "data_id": opening.get("data_id"),
        "conversations": [
            {"input": opening.get("input", [])},
            {"output": output_items},
        ],
    }
    output_json = model_output_dir / f"{total_uid}.json"
    output_json.write_text(json.dumps(record, indent=4, ensure_ascii=False), encoding="utf-8")
    return output_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert official OpenING JSONL to Q-CAGE JSONL")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--split-outputs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        convert_opening_file(
            input_jsonl=args.input_jsonl,
            output_jsonl=args.output_jsonl,
            split_outputs=args.split_outputs,
        )
    )


if __name__ == "__main__":
    main()

