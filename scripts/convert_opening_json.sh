#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:?Usage: bash scripts/convert_opening_json.sh INPUT_JSONL OUTPUT_JSONL [--split-outputs]}"
OUTPUT="${2:?Usage: bash scripts/convert_opening_json.sh INPUT_JSONL OUTPUT_JSONL [--split-outputs]}"
SPLIT="${3:-}"

ARGS=(--input-jsonl "${INPUT}" --output-jsonl "${OUTPUT}")
if [[ "${SPLIT}" == "--split-outputs" ]]; then
  ARGS+=(--split-outputs)
fi

python -m qcage.evaluation.opening_official "${ARGS[@]}"
