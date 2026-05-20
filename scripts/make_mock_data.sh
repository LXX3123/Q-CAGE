#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/mock_tiny.yaml}"

python -m qcage.data.make_mock_data \
  --config "${CONFIG}"

