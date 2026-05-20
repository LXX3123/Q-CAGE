from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class IntJudgeSummary:
    fdt_wins: float
    fdt_total: int
    non_tie_wins: int
    non_tie_total: int
    tie0_wins: int
    ties: int
    total: int

    def as_percentages(self) -> dict[str, float]:
        fdt = 100.0 * self.fdt_wins / self.fdt_total if self.fdt_total else 0.0
        without_tie = 100.0 * self.non_tie_wins / self.non_tie_total if self.non_tie_total else 0.0
        with_tie_zero = 100.0 * self.tie0_wins / self.total if self.total else 0.0
        with_tie_half = (
            100.0 * (self.tie0_wins + 0.5 * self.ties) / self.total if self.total else 0.0
        )
        return {
            "FDT": fdt,
            "w/o Tie": without_tie,
            "w/Tie (0)": with_tie_zero,
            "w/Tie (.5)": with_tie_half,
        }


def _model_name(record: dict, side: str) -> str | None:
    official_key = f"model_{side.upper()}"
    if isinstance(record.get(official_key), dict):
        return record[official_key].get("name")
    for key in [f"candidate_{side}", f"model_{side}", f"{side}_name"]:
        if record.get(key):
            return str(record[key])
    return None


def _winner_side(record: dict) -> str:
    winner = str(record.get("winner", record.get("judgement", record.get("label", "")))).strip()
    lowered = winner.lower()
    if lowered in {"a", "left", "candidate_a", "model_a", "1"}:
        return "A"
    if lowered in {"b", "right", "candidate_b", "model_b", "2"}:
        return "B"
    if lowered.startswith("tie(a"):
        return "Tie(A)"
    if lowered.startswith("tie(b"):
        return "Tie(B)"
    if lowered in {"tie", "draw", "equal", "same", "0"}:
        return "tie"
    return "other"


def summarize_records(records: Iterable[dict], candidate_name: str = "qcage") -> IntJudgeSummary:
    fdt_wins = 0.0
    fdt_total = 0
    non_tie_wins = 0
    non_tie_total = 0
    tie0_wins = 0
    ties = 0
    total = 0

    for record in records:
        model_a = _model_name(record, "a")
        model_b = _model_name(record, "b")
        raw_winner = str(record.get("winner", record.get("judgement", record.get("label", "")))).strip()
        winner = _winner_side(record)
        candidate_side = "A" if model_a == candidate_name else "B" if model_b == candidate_name else None
        if raw_winner == candidate_name:
            candidate_side = "candidate"
            winner = "candidate"
        if candidate_side is None:
            continue

        total += 1
        fdt_total += 1

        if winner in {"tie", "Tie(A)", "Tie(B)"}:
            ties += 1
        if winner in {"A", "B"}:
            non_tie_total += 1
            if winner == candidate_side or winner == "candidate":
                non_tie_wins += 1
                tie0_wins += 1
                fdt_wins += 1
        elif winner in {"Tie(A)", "Tie(B)"}:
            preferred_side = winner[4]
            if preferred_side == candidate_side:
                fdt_wins += 1
        elif winner == "candidate":
            non_tie_total += 1
            non_tie_wins += 1
            tie0_wins += 1
            fdt_wins += 1

    return IntJudgeSummary(
        fdt_wins=fdt_wins,
        fdt_total=fdt_total,
        non_tie_wins=non_tie_wins,
        non_tie_total=non_tie_total,
        tie0_wins=tie0_wins,
        ties=ties,
        total=total,
    )


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_judgements(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return load_csv(path)
    return load_jsonl(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize IntJudge pairwise results")
    parser.add_argument("--judgements", required=True, help="JSONL/CSV with winner field")
    parser.add_argument("--candidate-name", default="qcage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_records(load_judgements(args.judgements), candidate_name=args.candidate_name)
    print(json.dumps({"counts": summary.__dict__, "metrics": summary.as_percentages()}, indent=2))


if __name__ == "__main__":
    main()
