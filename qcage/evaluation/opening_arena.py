from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from qcage.evaluation.opening_official import load_opening_jsonl


def _flatten_models(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    models: list[dict[str, str]] = []
    for index, item in enumerate(items):
        key = str(index)
        value = item.get(key)
        if value is None and len(item) == 1:
            key, value = next(iter(item.items()))
        if value is None:
            continue
        models.append({"id": str(key), "name": value["name"], "github": value.get("github", "")})
    return models


def _pack_models(models: list[dict[str, str]]) -> list[dict[str, dict[str, str]]]:
    return [
        {str(index): {"name": model["name"], "github": model.get("github", "")}}
        for index, model in enumerate(models)
    ]


def ensure_model(
    *,
    baseline_models_json: str | Path,
    model_name: str,
    github: str = "",
) -> dict[str, str]:
    path = Path(baseline_models_json)
    items = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    models = _flatten_models(items)
    for model in models:
        if model["name"] == model_name:
            return model
    models.append({"id": str(len(models)), "name": model_name, "github": github})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_pack_models(models), indent=4, ensure_ascii=False), encoding="utf-8")
    return models[-1]


def make_pairs(
    *,
    benchmark_jsonl: str | Path,
    baseline_models_json: str | Path,
    output_json: str | Path,
    target_model: str,
    opponents: list[str] | None = None,
    seed: int = 1234,
    target_side: str = "alternate",
) -> Path:
    rng = random.Random(seed)
    models = _flatten_models(json.loads(Path(baseline_models_json).read_text(encoding="utf-8")))
    by_name = {model["name"]: model for model in models}
    if target_model not in by_name:
        raise KeyError(f"{target_model} is not present in {baseline_models_json}")
    if opponents is None or not opponents:
        opponents = [name for name in by_name if name != target_model]

    records = load_opening_jsonl(benchmark_jsonl)
    pairs: list[dict[str, Any]] = []
    for row_index, record in enumerate(records):
        for opponent in opponents:
            if opponent not in by_name:
                raise KeyError(f"Opponent {opponent} is not present in {baseline_models_json}")
            target = by_name[target_model]
            other = by_name[opponent]
            if target_side == "A":
                model_a, model_b = target, other
            elif target_side == "B":
                model_a, model_b = other, target
            elif target_side == "random":
                model_a, model_b = (target, other) if rng.random() < 0.5 else (other, target)
            else:
                model_a, model_b = (target, other) if row_index % 2 == 0 else (other, target)
            pairs.append(
                {
                    "data_id": record["total_uid"],
                    "model_A": {"id": model_a["id"], "name": model_a["name"]},
                    "model_B": {"id": model_b["id"], "name": model_b["name"]},
                }
            )

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(pairs, indent=4, ensure_ascii=False), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OpenING Arena model list and pair files")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    add_model = subparsers.add_parser("ensure-model")
    add_model.add_argument("--baseline-models-json", required=True)
    add_model.add_argument("--model-name", required=True)
    add_model.add_argument("--github", default="")

    pairs = subparsers.add_parser("make-pairs")
    pairs.add_argument("--benchmark-jsonl", required=True)
    pairs.add_argument("--baseline-models-json", required=True)
    pairs.add_argument("--output-json", required=True)
    pairs.add_argument("--target-model", required=True)
    pairs.add_argument("--opponents", nargs="*", default=None)
    pairs.add_argument("--seed", type=int, default=1234)
    pairs.add_argument("--target-side", choices=["A", "B", "alternate", "random"], default="alternate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "ensure-model":
        print(
            json.dumps(
                ensure_model(
                    baseline_models_json=args.baseline_models_json,
                    model_name=args.model_name,
                    github=args.github,
                ),
                indent=2,
            )
        )
    elif args.cmd == "make-pairs":
        print(
            make_pairs(
                benchmark_jsonl=args.benchmark_jsonl,
                baseline_models_json=args.baseline_models_json,
                output_json=args.output_json,
                target_model=args.target_model,
                opponents=args.opponents,
                seed=args.seed,
                target_side=args.target_side,
            )
        )


if __name__ == "__main__":
    main()

