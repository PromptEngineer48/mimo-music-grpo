#!/usr/bin/env bash
# Does the published model keep its gains? merged bf16 (vLLM) vs Q8 vs Q4 GGUF (llama.cpp), same 40 prompts.
set -euo pipefail
cd /workspace/music-demo
export THINK=0 MAXLEN=6144
REPO=Prompt48/Qwen3-8B-MiMo-Music-GRPO

# 1. merged model from HF, same eval as the original "after" run
python generate.py --model $REPO --tag hf_merged --k 4 --max_tokens 4096

# 2. llama.cpp with CUDA
[ -d /workspace/llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp
pip install -q cmake
cmake -S /workspace/llama.cpp -B /workspace/llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF > /dev/null
cmake --build /workspace/llama.cpp/build --target llama-server -j 32 > /dev/null
python -c "from huggingface_hub import hf_hub_download as d
for q in ('Q4_K_M','Q8_0'): print(d('$REPO-GGUF', f'qwen3-8b-mimo-music-grpo-{q}.gguf', local_dir='/workspace/gguf'))"

for q in Q8_0 Q4_K_M; do
  /workspace/llama.cpp/build/bin/llama-server -m /workspace/gguf/qwen3-8b-mimo-music-grpo-$q.gguf \
      -ngl 99 -c 65536 -np 8 --port 8080 > /workspace/llama_$q.log 2>&1 &
  SP=$!
  until curl -sf http://127.0.0.1:8080/health > /dev/null; do sleep 3; done
  python gguf_eval.py --tag gguf_$q --k 4
  kill $SP; sleep 5
done
echo QUANT-DONE
