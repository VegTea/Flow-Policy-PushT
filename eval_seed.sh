#!/usr/bin/env bash
set -euo pipefail

source ~/.local/bin/env
cd /inspire/ssd/project/gjjproject/czxs24230043/Flow-Policy-PushT

CKPT="data/outputs/2026.06.15/12.39.16_train_flow_matching_unet_hybrid_pusht_image/checkpoints/epoch=0400-test_mean_score=0.869.ckpt"
LOG_DIR="$(dirname "$(dirname "${CKPT}")")"           # data/outputs/2026.06.15/12.39.16_train_flow_matching_unet_hybrid_pusht_image
CKPT_NAME="$(basename "${CKPT}" .ckpt)"                # epoch=0400-test_mean_score=0.869
NUM_INFERENCE_STEPS=40
POLICY_SEED=0

uv run --frozen python eval_seeds.py \
    --checkpoint "${CKPT}" \
    --output_dir "${LOG_DIR}/eval_output/${CKPT_NAME}_40steps" \
    --seeds 100012,100013,100022,100025,100026,100027,100041,100049 \
    --device cuda:0 \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --policy_seed "${POLICY_SEED}"
