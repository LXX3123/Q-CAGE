from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qcage.evaluation.intjudge_metrics import summarize_records


class MetricsTest(unittest.TestCase):
    def test_tie_aware_metrics(self) -> None:
        records = [
            {"model_A": {"name": "qcage"}, "model_B": {"name": "baseline"}, "winner": "A"},
            {"model_A": {"name": "qcage"}, "model_B": {"name": "baseline"}, "winner": "B"},
            {"model_A": {"name": "qcage"}, "model_B": {"name": "baseline"}, "winner": "Tie(A)"},
            {"model_A": {"name": "baseline"}, "model_B": {"name": "qcage"}, "winner": "Tie(A)"},
        ]
        summary = summarize_records(records, candidate_name="qcage")
        metrics = summary.as_percentages()
        self.assertEqual(summary.tie0_wins, 1)
        self.assertEqual(summary.non_tie_total, 2)
        self.assertEqual(summary.ties, 2)
        self.assertAlmostEqual(metrics["FDT"], 50.0)
        self.assertAlmostEqual(metrics["w/o Tie"], 50.0)
        self.assertAlmostEqual(metrics["w/Tie (0)"], 25.0)
        self.assertAlmostEqual(metrics["w/Tie (.5)"], 50.0)


if __name__ == "__main__":
    unittest.main()
