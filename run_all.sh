#!/usr/bin/env bash
# The exact run shown in the video. ~2.5h on 1x H100 80GB.
#   bash setup_pod.sh && bash run_all.sh
set -euo pipefail
export THINK=0 MAXLEN=6144          # Qwen3 thinking OFF (short answers, fast training)

# 1. BEFORE: base model on the 40 held-out prompts, 4 samples each
python generate.py --model Qwen/Qwen3-8B --tag before --k 4 --max_tokens 4096

# 2. TRAIN: GRPO, 30 steps x (8 prompts x 8 answers), LoRA r=64   (~1.7h)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train_grpo.py --max_steps 30 --prompts_per_step 8 --lr 2e-5

# 3. AFTER: trained model, same prompts, same seed
python generate.py --model ckpt/music-grpo/final --tag after --k 4 --max_tokens 4096

# 4. Media for the video
python render.py --tag before
python render.py --tag after
python charts.py
python blind_fair.py                # fair A/B pairs -> blind/index.html
echo DONE
