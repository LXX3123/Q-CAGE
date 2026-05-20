from __future__ import annotations

from qcage.evaluation.intjudge_metrics import IntJudgeSummary


def format_metrics_table(summary: IntJudgeSummary) -> str:
    metrics = summary.as_percentages()
    lines = ["metric,value"]
    for key, value in metrics.items():
        lines.append(f"{key},{value:.2f}")
    return "\n".join(lines)

