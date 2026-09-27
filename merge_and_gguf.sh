#!/usr/bin/env bash
# On a GPU pod: merge LoRA into Qwen3-8B, build GGUFs, push both repos to HF.
# Needs: /workspace/ckpt/music-grpo/adapter, HF token in ~/.cache/huggingface/token
set -euo pipefail
cd /workspace
REPO=Prompt48/Qwen3-8B-MiMo-Music-GRPO
GGUF_REPO=Prompt48/Qwen3-8B-MiMo-Music-GRPO-GGUF

pip install -q peft "transformers>=4.51" accelerate huggingface_hub sentencepiece gguf

[ -f merged/config.json ] || python - <<'EOF'
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base, "ckpt/music-grpo/adapter").merge_and_unload()
model.save_pretrained("merged", safe_serialization=True, max_shard_size="5GB")
AutoTokenizer.from_pretrained("Qwen/Qwen3-8B").save_pretrained("merged")  # same tokenizer as base
print("MERGED-OK")
EOF

# tokenizer identical to base; use base files (adapter copy was saved by transformers 5.x)
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-8B').save_pretrained('merged')"
[ -d llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp
pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
mkdir -p gguf
python llama.cpp/convert_hf_to_gguf.py merged --outtype bf16 --outfile gguf/qwen3-8b-mimo-music-grpo-BF16.gguf
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > /dev/null
cmake --build llama.cpp/build --target llama-quantize -j 16 > /dev/null
for q in Q4_K_M Q8_0; do
  llama.cpp/build/bin/llama-quantize gguf/qwen3-8b-mimo-music-grpo-BF16.gguf gguf/qwen3-8b-mimo-music-grpo-$q.gguf $q > /dev/null
done
rm gguf/qwen3-8b-mimo-music-grpo-BF16.gguf
ls -la gguf
echo GGUF-OK

python - <<EOF
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("$REPO", exist_ok=True)
api.upload_folder(folder_path="merged", repo_id="$REPO", commit_message="Merged Qwen3-8B + MiMo music GRPO LoRA")
api.upload_folder(folder_path="ckpt/music-grpo/adapter", path_in_repo="lora_adapter", repo_id="$REPO", commit_message="LoRA adapter")
api.upload_file(path_or_fileobj="README_model.md", path_in_repo="README.md", repo_id="$REPO")
api.create_repo("$GGUF_REPO", exist_ok=True)
api.upload_folder(folder_path="gguf", repo_id="$GGUF_REPO", commit_message="GGUF Q4_K_M + Q8_0")
api.upload_file(path_or_fileobj="README_gguf.md", path_in_repo="README.md", repo_id="$GGUF_REPO")
print("PUSH-OK")
EOF
echo ALL-DONE
