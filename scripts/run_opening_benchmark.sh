#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/benchmark_opening.yaml}"
GPUS="${2:-8}"

torchrun --standalone --nproc_per_node="${GPUS}" -m qcage.evaluation.run_benchmark \
  --config "${CONFIG}"

