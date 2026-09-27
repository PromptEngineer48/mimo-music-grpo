"""Blind A/B listening test -- the proof the grader can't fake.

    python blind_test.py --n 10

Picks N held-out prompts where BOTH versions produced playable audio, randomly
labels them A/B, copies to blind/, and writes:
  blind/index.html   -- open in browser (or screen-record for the video)
  blind/ANSWER_KEY.json  -- keep hidden until the reveal
Picking is by prompt order, not by score, so it is not cherry-picked.
"""

import argparse
import json
import random
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    before, after = Path("media/before"), Path("media/after")
    ids = sorted(p.stem for p in after.glob("*.wav") if (before / f"{p.stem}.wav").exists())
    if len(ids) < args.n:
        print(f"only {len(ids)} prompts have audio in both; using all")
    ids = ids[: args.n]
    prompts = {r["src_id"]: r["prompt"] for r in json.load(open("results/after.json", encoding="utf-8"))}

    rng = random.Random(args.seed)
    out = Path("blind")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    key, rows = {}, []
    for i, sid in enumerate(ids, 1):
        flip = rng.random() < 0.5
        a, b = (after, before) if flip else (before, after)
        shutil.copy(a / f"{sid}.wav", out / f"{i:02d}_A.wav")
        shutil.copy(b / f"{sid}.wav", out / f"{i:02d}_B.wav")
        key[f"{i:02d}"] = {"trained": "A" if flip else "B", "src_id": sid}
        rows.append(f"""<div class=q><h3>Pair {i}</h3><p>{prompts.get(sid, '')}</p>
<div class=ab><span>A</span><audio controls src="{i:02d}_A.wav"></audio></div>
<div class=ab><span>B</span><audio controls src="{i:02d}_B.wav"></audio></div></div>""")

    json.dump(key, open(out / "ANSWER_KEY.json", "w"), indent=1)
    (out / "index.html").write_text(f"""<!doctype html><meta charset=utf-8><title>Blind Test</title>
<style>body{{background:#0b0b12;color:#fff;font-family:system-ui;max-width:820px;margin:auto;padding:24px}}
.q{{border:1px solid #333;border-radius:12px;padding:16px;margin:16px 0}}p{{color:#aaa}}
.ab{{display:flex;gap:12px;align-items:center;margin:8px 0}}.ab span{{font-size:28px;font-weight:700;color:#ff7a18;width:30px}}
audio{{width:100%}}</style><h1>Which one sounds better? A or B?</h1>{''.join(rows)}""", encoding="utf-8")
    print(f"{len(ids)} pairs -> blind/index.html  (answer key: blind/ANSWER_KEY.json)")


if __name__ == "__main__":
    main()
