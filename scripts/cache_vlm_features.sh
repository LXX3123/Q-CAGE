#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/train_a100_bf16.yaml}"
OUT="${2:-cached_features}"
OUT_JSONL="${3:-}"

if [[ -n "${OUT_JSONL}" ]]; then
  python -m qcage.training.cache_vlm_features \
    --config "${CONFIG}" \
    --output-dir "${OUT}" \
    --output-jsonl "${OUT_JSONL}"
else
  python -m qcage.training.cache_vlm_features \
    --config "${CONFIG}" \
    --output-dir "${OUT}"
fi
