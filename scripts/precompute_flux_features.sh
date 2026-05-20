#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/train_flux_precomputed_a100.yaml}"
INPUT_JSONL="${2:-data/train_with_qwen_features.jsonl}"
OUTPUT_DIR="${3:-cached_features_flux}"
OUTPUT_JSONL="${4:-data/train_precomputed_flux.jsonl}"

python -m qcage.training.precompute_flux_features \
  --config "${CONFIG}" \
  --input-jsonl "${INPUT_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-jsonl "${OUTPUT_JSONL}"

