source ~/.local/bin/env

uv run --frozen python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.resume=true \
  training.device=cuda:0 \
  hydra.run.dir=data/outputs/2026.06.15/01.52.29_train_diffusion_unet_hybrid_pusht_image