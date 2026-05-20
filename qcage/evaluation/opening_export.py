from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_pairwise_manifest(
    *,
    candidate_manifest: str | Path,
    baseline_manifest: str | Path,
    output_jsonl: str | Path,
    candidate_name: str = "qcage",
    baseline_name: str = "baseline",
) -> Path:
    candidate_records = _load_by_id(candidate_manifest)
    baseline_records = _load_by_id(baseline_manifest)
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for sample_id, candidate in candidate_records.items():
            if sample_id not in baseline_records:
                continue
            pair = {
                "sample_id": sample_id,
                "candidate_a": candidate_name,
                "candidate_b": baseline_name,
                "candidate_a_output": candidate.get("generation"),
                "candidate_b_output": baseline_records[sample_id].get("generation"),
                "winner": None,
            }
            handle.write(json.dumps(pair) + "\n")
    return output_path


def _load_by_id(path: str | Path) -> dict[str, dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return {record["sample_id"]: record for record in records}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export pairwise manifest for IntJudge")
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--candidate-name", default="qcage")
    parser.add_argument("--baseline-name", default="baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = export_pairwise_manifest(
        candidate_manifest=args.candidate_manifest,
        baseline_manifest=args.baseline_manifest,
        output_jsonl=args.output_jsonl,
        candidate_name=args.candidate_name,
        baseline_name=args.baseline_name,
    )
    print(path)


if __name__ == "__main__":
    main()

