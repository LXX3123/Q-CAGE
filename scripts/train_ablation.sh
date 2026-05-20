#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?Usage: bash scripts/train_ablation.sh CONFIG [GPUS]}"
GPUS="${2:-8}"

torchrun --standalone --nproc_per_node="${GPUS}" -m qcage.training.train_adapter \
  --config "${CONFIG}"

