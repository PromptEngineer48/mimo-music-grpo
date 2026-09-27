"""Generate + score pieces on the HELD-OUT prompts, for any model.

    python generate.py --model Qwen/Qwen2.5-3B-Instruct --tag before
    python generate.py --model ckpt/music-grpo/final      --tag after

Writes results/<tag>.json : per prompt, k samples with ABC, score, group breakdown.
Same seed + same prompts for both tags -> fair comparison.
"""

import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams

from reward import score_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=4, help="samples per prompt")
    ap.add_argument("--max_tokens", type=int, default=4096)
    args = ap.parse_args()

    test = [json.loads(l) for l in open("data/test.jsonl", encoding="utf-8")]
    llm = LLM(model=args.model, seed=48, gpu_memory_utilization=0.85, max_model_len=int(__import__("os").environ.get("MAXLEN","6144")))
    sp = SamplingParams(n=args.k, temperature=1.0, max_tokens=args.max_tokens, seed=48)
    import os
    ctk = {"enable_thinking": os.environ.get("THINK", "1") == "1"}
    test = test[: int(os.environ.get("NPROMPTS", len(test)))]
    outs = llm.chat([t["prompt"] for t in test], sp, chat_template_kwargs=ctk)

    results = []
    for t, o in zip(test, outs):
        samples = []
        for c in o.outputs:
            s = score_one(c.text)
            samples.append({"text": c.text, "n_tokens": len(c.token_ids), **s})
        results.append({"src_id": t["src_id"], "tag": t["tag"], "prompt": t["prompt"][-1]["content"], "samples": samples})

    Path("results").mkdir(exist_ok=True)
    json.dump(results, open(f"results/{args.tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    flat = [s for r in results for s in r["samples"]]
    valid = sum(s["total"] > 0 for s in flat) / len(flat)
    mean = sum(s["total"] for s in flat) / len(flat)
    print(f"[{args.tag}] samples={len(flat)} valid_music={valid:.1%} mean_score={mean:.3f}")


if __name__ == "__main__":
    main()
