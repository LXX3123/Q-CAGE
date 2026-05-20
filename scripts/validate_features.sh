#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/train_flux_precomputed_a100.yaml}"
JSONL="${2:-}"
REQUIRE_FLUX="${3:-1}"

ARGS=(--config "${CONFIG}")
if [[ -n "${JSONL}" ]]; then
  ARGS+=(--input-jsonl "${JSONL}")
fi
if [[ "${REQUIRE_FLUX}" == "1" ]]; then
  ARGS+=(--require-flux)
fi

python -m qcage.data.validate_features "${ARGS[@]}"

