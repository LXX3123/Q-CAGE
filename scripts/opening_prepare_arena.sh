#!/usr/bin/env bash
set -euo pipefail

OPENING_ROOT="${1:?Usage: bash scripts/opening_prepare_arena.sh OPENING_ROOT MODEL_NAME BENCHMARK_JSONL PAIRS_JSON [OPPONENT...]}"
MODEL_NAME="${2:?Usage: bash scripts/opening_prepare_arena.sh OPENING_ROOT MODEL_NAME BENCHMARK_JSONL PAIRS_JSON [OPPONENT...]}"
BENCHMARK_JSONL="${3:?Usage: bash scripts/opening_prepare_arena.sh OPENING_ROOT MODEL_NAME BENCHMARK_JSONL PAIRS_JSON [OPPONENT...]}"
PAIRS_JSON="${4:?Usage: bash scripts/opening_prepare_arena.sh OPENING_ROOT MODEL_NAME BENCHMARK_JSONL PAIRS_JSON [OPPONENT...]}"
shift 4

BASELINE_JSON="${OPENING_ROOT}/Interleaved_Arena/baseline_models.json"

python -m qcage.evaluation.opening_arena ensure-model \
  --baseline-models-json "${BASELINE_JSON}" \
  --model-name "${MODEL_NAME}"

python -m qcage.evaluation.opening_arena make-pairs \
  --benchmark-jsonl "${BENCHMARK_JSONL}" \
  --baseline-models-json "${BASELINE_JSON}" \
  --output-json "${PAIRS_JSON}" \
  --target-model "${MODEL_NAME}" \
  --opponents "$@"

