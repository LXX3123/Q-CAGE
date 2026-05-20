#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/train_a100_bf16.yaml}"
GPUS="${2:-8}"

torchrun --standalone --nproc_per_node="${GPUS}" -m qcage.training.train_adapter \
  --config "${CONFIG}"

