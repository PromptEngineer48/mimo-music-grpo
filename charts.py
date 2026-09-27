"""Video charts: before vs after, plus the live training curve.

    python charts.py --before results/before.json --after results/after.json \
                     --trainer_state ckpt/music-grpo/checkpoint-*/trainer_state.json

Writes charts/*.png:
  1_valid_rate.png      % of outputs that are real, playable music
  2_score_hist.png      score distribution before vs after
  3_radar.png           the 6 things Xiaomi's grader measures, before vs after
  4_reward_curve.png    reward during training (from TRL logs)
  5_length.png          answer length before vs after (length-penalty effect)
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, FG, C_BEFORE, C_AFTER = "#0b0b12", "white", "#888899", "#ff7a18"
GROUPS = ["rhythm", "texture", "acoustic", "tonal", "register", "structure"]
plt.rcParams.update({"text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                     "axes.facecolor": BG, "figure.facecolor": BG, "font.size": 13})


def flat(path):
    return [s for r in json.load(open(path, encoding="utf-8")) for s in r["samples"]]


def save(fig, name):
    Path("charts").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"charts/{name}", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="results/before.json")
    ap.add_argument("--after", default="results/after.json")
    ap.add_argument("--trainer_state", default="ckpt/music-grpo/checkpoint-*/trainer_state.json")
    args = ap.parse_args()
    B, A = flat(args.before), flat(args.after)

    # 1. valid rate
    vb, va = np.mean([s["total"] > 0 for s in B]) * 100, np.mean([s["total"] > 0 for s in A]) * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["Before RL", "After RL"], [vb, va], color=[C_BEFORE, C_AFTER])
    ax.bar_label(bars, fmt="%.1f%%", color=FG, fontsize=18)
    ax.set_ylim(0, 110)
    ax.set_title("Outputs that are real, playable music")
    save(fig, "1_valid_rate.png")

    # 2. score histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist([s["total"] for s in B], bins, alpha=0.7, color=C_BEFORE, label=f"Before (mean {np.mean([s['total'] for s in B]):.2f})")
    ax.hist([s["total"] for s in A], bins, alpha=0.7, color=C_AFTER, label=f"After (mean {np.mean([s['total'] for s in A]):.2f})")
    ax.set_xlabel("Xiaomi 'human-likeness' score")
    ax.set_ylabel("# pieces")
    ax.legend(facecolor=BG)
    save(fig, "2_score_hist.png")

    # 3. radar over valid pieces (groups are 0..100 each)
    def gmean(S):
        v = [s for s in S if s.get("groups")]
        return [np.mean([s["groups"].get(g, 0) for s in v]) if v else 0 for g in GROUPS]

    ang = np.linspace(0, 2 * np.pi, len(GROUPS), endpoint=False).tolist()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(polar=True)
    ax.set_facecolor(BG)
    for vals, col, lab in ((gmean(B), C_BEFORE, "Before"), (gmean(A), C_AFTER, "After")):
        ax.plot(ang + ang[:1], vals + vals[:1], color=col, lw=3, label=lab)
        ax.fill(ang + ang[:1], vals + vals[:1], color=col, alpha=0.2)
    ax.set_xticks(ang)
    ax.set_xticklabels([g.title() for g in GROUPS])
    ax.legend(loc="upper right", facecolor=BG)
    ax.set_title("What improved (valid pieces only)")
    save(fig, "3_radar.png")

    # 4. reward curve
    states = sorted(glob.glob(args.trainer_state))
    if states:
        log = json.load(open(states[-1]))["log_history"]
        steps = [e["step"] for e in log if "reward" in e]
        rew = [e["reward"] for e in log if "reward" in e]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, rew, color=C_AFTER, alpha=0.35)
        k = max(1, len(rew) // 15)
        ax.plot(steps[k - 1:], np.convolve(rew, np.ones(k) / k, "valid"), color=C_AFTER, lw=3)
        ax.set_xlabel("training step")
        ax.set_ylabel("mean reward")
        ax.set_title("Reward during GRPO training")
        save(fig, "4_reward_curve.png")

    # 5. length
    fig, ax = plt.subplots(figsize=(7, 5))
    lb, la = np.mean([s["n_tokens"] for s in B]), np.mean([s["n_tokens"] for s in A])
    bars = ax.bar(["Before RL", "After RL"], [lb, la], color=[C_BEFORE, C_AFTER])
    ax.bar_label(bars, fmt="%.0f", color=FG, fontsize=18)
    ax.set_title("Average answer length (tokens)")
    save(fig, "5_length.png")
    print(f"valid {vb:.0f}% -> {va:.0f}% | charts/ written")


if __name__ == "__main__":
    main()
