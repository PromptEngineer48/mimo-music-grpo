"""Score a GGUF (served by llama-server) on the same 40 held-out prompts as generate.py.

    llama-server -m model.gguf -ngl 99 -c 8192 --port 8080 &
    python gguf_eval.py --tag gguf_q4 --k 4

Uses the exact training chat format (thinking off = empty <think></think>) and the
same sampling as the vLLM eval (temperature 1.0, no top-k/top-p/repeat penalty).
"""

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from reward import score_one


def prompt_text(msgs):
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
    return s + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def complete(port, text, seed, max_tokens):
    body = {"prompt": text, "n_predict": max_tokens, "temperature": 1.0, "top_k": 0, "top_p": 1.0,
            "min_p": 0.0, "repeat_penalty": 1.0, "seed": seed, "stop": ["<|im_end|>"], "cache_prompt": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return r["content"], r.get("tokens_predicted", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()

    test = [json.loads(l) for l in open("data/test.jsonl", encoding="utf-8")]
    jobs = [(t, i) for t in test for i in range(args.k)]

    def run(job):
        t, i = job
        text, n = complete(args.port, prompt_text(t["prompt"]), 48 + i, args.max_tokens)
        return t["src_id"], {"text": text, "n_tokens": n, **score_one(text)}

    with ThreadPoolExecutor(args.parallel) as ex:
        outs = list(ex.map(run, jobs))
    by = {}
    for sid, s in outs:
        by.setdefault(sid, []).append(s)
    results = [{"src_id": t["src_id"], "tag": t["tag"], "prompt": t["prompt"][-1]["content"], "samples": by[t["src_id"]]} for t in test]
    Path("results").mkdir(exist_ok=True)
    json.dump(results, open(f"results/{args.tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    flat = [s for r in results for s in r["samples"]]
    print(f"[{args.tag}] samples={len(flat)} valid_music={sum(s['total'] > 0 for s in flat) / len(flat):.1%} "
          f"mean_score={sum(s['total'] for s in flat) / len(flat):.3f} avg_tokens={sum(s['n_tokens'] for s in flat) // len(flat)}")


if __name__ == "__main__":
    main()
