#!/usr/bin/env bash
set -euo pipefail

MODEL_OUTPUT_DIR="${1:?Usage: bash scripts/opening_install_outputs.sh MODEL_OUTPUT_DIR OPENING_ROOT}"
OPENING_ROOT="${2:?Usage: bash scripts/opening_install_outputs.sh MODEL_OUTPUT_DIR OPENING_ROOT}"

DEST="${OPENING_ROOT}/Interleaved_Arena/$(basename "${MODEL_OUTPUT_DIR}")"
mkdir -p "${DEST}"
cp -r "${MODEL_OUTPUT_DIR}/." "${DEST}/"
echo "${DEST}"
