#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/infer_qcage.yaml}"

python -m qcage.inference.run_trajectory \
  --config "${CONFIG}"

