#!/usr/bin/env bash
set -euo pipefail

DEFAULT_SEEDS=()
for seed in $(seq 0 99); do
  DEFAULT_SEEDS+=("${seed}")
done
if [[ $# -gt 0 ]]; then
  SEEDS=("$@")
else
  # Reduce from the command line with, for example: SEEDS="0 1 2" bash scripts/run_albert_rte_reproduction_seeds.sh
  read -r -a SEEDS <<< "${SEEDS:-${DEFAULT_SEEDS[*]}}"
fi

HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME:-${HOME}/.cache/huggingface}}"
COMMON=(
  python train.py
  --config configs/albert_rte_reproduction.yaml
  --hf-cache-dir "${HF_CACHE_DIR}"
  --no-download
  --resume
)

run_one() {
  local optimizer_name="$1"
  local clip_threshold="$2"
  local seed="$3"
  "${COMMON[@]}" \
    --optimizer_name "${optimizer_name}" \
    --clip_threshold "${clip_threshold}" \
    --seed "${seed}"
}

for seed in "${SEEDS[@]}"; do
  run_one adamw_uncut inf "${seed}"
  run_one adamw_clip 2.0 "${seed}"
  run_one adamw_resclip_euclidean 2.0 "${seed}"
done
