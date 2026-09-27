#!/usr/bin/env bash
# One-time setup on a fresh RunPod / Linux GPU box (PyTorch template, CUDA 12.x).
set -euo pipefail

apt-get update -qq
apt-get install -y -qq abcmidi fluidsynth fluid-soundfont-gm ffmpeg

pip install -q "trl>=0.23" "vllm>=0.10" datasets pandas pyarrow matplotlib pretty_midi wandb

# Xiaomi's music prompts (1000 prompts; we use the 496 English ones)
mkdir -p data
[ -f data/music.parquet ] || wget -q -O data/music.parquet \
  https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss/resolve/main/music.parquet

python prep_data.py
python precheck.py
echo "setup done. optional: wandb login"
