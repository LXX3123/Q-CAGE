#!/usr/bin/env bash
set -euo pipefail

JUDGEMENTS="${1:?Usage: bash scripts/summarize_intjudge.sh JUDGEMENTS_JSONL_OR_CSV [CANDIDATE_NAME]}"
CANDIDATE="${2:-qcage}"

python -m qcage.evaluation.intjudge_metrics \
  --judgements "${JUDGEMENTS}" \
  --candidate-name "${CANDIDATE}"

