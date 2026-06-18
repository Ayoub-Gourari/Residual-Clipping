#!/usr/bin/env bash
set -euo pipefail

EXTRA_ARGS=("$@")
HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME:-${HOME}/.cache/huggingface}}"
COMMON=(
  python train.py
  --config configs/albert_rte_reproduction.yaml
  --hf-cache-dir "${HF_CACHE_DIR}"
  --no-download
  --seed 123
  --resume
)

run_one() {
  local optimizer_name="$1"
  local clip_threshold="$2"
  "${COMMON[@]}" \
    --optimizer_name "${optimizer_name}" \
    --clip_threshold "${clip_threshold}" \
    "${EXTRA_ARGS[@]}"
}

run_one adamw_uncut inf
run_one adamw_clip 1.0
run_one adamw_resclip_euclidean 1.0
