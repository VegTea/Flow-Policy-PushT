#!/usr/bin/env bash
set -euo pipefail

source ~/.local/bin/env
cd /inspire/ssd/project/gjjproject/czxs24230043/Flow-Policy-PushT

uv run --frozen python train.py \
  --config-dir=. \
  --config-name=image_pusht_flow_matching_unet_cnn.yaml \
  training.seed=42 \
  training.device=cuda:0 \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
