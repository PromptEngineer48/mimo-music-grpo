"""Split Xiaomi's music prompts into train / held-out test.

Uses only English prompts by default (so the audience can read them) and keeps
a fixed held-out set the model NEVER trains on -- that set is what we show on video.

    python prep_data.py            # writes data/train.jsonl, data/test.jsonl
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

SEED = 48
SYSTEM = (
    "You are a composer. Reply with one complete piece in ABC notation inside a "
    "```abc code block. Start with X:1 and include T:, M:, L:, Q:, K: headers."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/music.parquet")
    ap.add_argument("--lang", default="en", help="en | zh | all")
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--short_only", action="store_true",
                    help="keep only 'short' pieces (note: in this dataset all short ones are zh)")
    args = ap.parse_args()

    df = pd.read_parquet(args.src)
    rows = []
    for _, r in df.iterrows():
        info = dict(r["extra_info"])
        if args.lang != "all" and info.get("lang") != args.lang:
            continue
        if args.short_only and info.get("length") != "short":
            continue
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": r["prompt"][0]["content"]},
            ],
            "src_id": info.get("src_id", ""),
            "tag": info.get("tag", ""),
            "bpm": int(info.get("bpm") or 0),
            "meter": info.get("meter", ""),
            "nvoice_want": int(info.get("nvoice_want") or 0),
        })

    random.Random(SEED).shuffle(rows)
    test, train = rows[: args.n_test], rows[args.n_test :]
    out = Path("data")
    out.mkdir(exist_ok=True)
    for name, split in (("train", train), ("test", test)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in split:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"train={len(train)} test={len(test)} (lang={args.lang})")


if __name__ == "__main__":
    main()
