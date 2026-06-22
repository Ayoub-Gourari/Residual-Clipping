#!/usr/bin/env bash
set -euo pipefail

EXTRA_ARGS=("$@")
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-albert-base-v2}"
MODEL_REVISION="${MODEL_REVISION:-8e2f239}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME:-${HOME}/.cache/huggingface}}"
SEEDS=(123 0 1)
THRESHOLDS=(0.1 0.2 0.5 1.0 2.0 5.0 10.0)

COMMON=(
  python train.py
  --task rte
  --model_checkpoint "${MODEL_CHECKPOINT}"
  --model-revision "${MODEL_REVISION}"
  --hf-cache-dir "${HF_CACHE_DIR}"
  --max_epochs 1
  --batch_size 8
  --eval_batch_size 8
  --learning_rate 1e-5
  --betas 0.9,0.999
  --eps 1e-6
  --weight_decay 0.0
  --correct_bias true
  --warmup_ratio 0.1
  --classifier_dropout 0.0
  --val_check_interval 12
  --clipping_scope local
  --save_checkpoints false
  --save_final_model false
  --wandb_log_model false
  --resume
)

run_one() {
  local optimizer_name="$1"
  local clip_threshold="$2"
  local seed="$3"
  "${COMMON[@]}" \
    --optimizer_name "${optimizer_name}" \
    --clip_threshold "${clip_threshold}" \
    --seed "${seed}" \
    "${EXTRA_ARGS[@]}"
}

for seed in "${SEEDS[@]}"; do
  run_one adamw_uncut inf "${seed}"
  for optimizer_name in adamw_clip adamw_resclip_euclidean adamw_resclip_euclidean_vclip adamw_resclip_euclidean_vclip_varalpha adamw_resclip_metric adamw_resclip_metric_vclip ResidualClipAdamW-M; do
    for clip_threshold in "${THRESHOLDS[@]}"; do
      run_one "${optimizer_name}" "${clip_threshold}" "${seed}"
    done
  done
done
