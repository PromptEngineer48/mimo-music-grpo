"""Turn results/<tag>.json into listenable + watchable files.

    python render.py --tag before
    python render.py --tag after

For each held-out prompt, takes SAMPLE #0 (not the best one -- no cherry picking)
and writes media/<tag>/<src_id>.{abc,mid,wav,png}. The PNG is a piano roll.
Invalid ABC gets a red "NO VALID MUSIC" card instead.
"""

import argparse
import json
import os
import subprocess as sp
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pretty_midi

SF2 = os.environ.get("SOUNDFONT", "/usr/share/sounds/sf2/FluidR3_GM.sf2")
ABC2MIDI = os.environ.get("ABC2MIDI_BIN", "abc2midi")


def piano_roll(mid_path, png_path, title):
    pm = pretty_midi.PrettyMIDI(str(mid_path))
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="#0b0b12")
    ax.set_facecolor("#0b0b12")
    colors = ["#ff7a18", "#35c7ff", "#b36bff", "#3ddc97", "#ffd166", "#ef476f"]
    for i, inst in enumerate(pm.instruments):
        for n in inst.notes:
            ax.add_patch(plt.Rectangle((n.start, n.pitch - 0.4), n.end - n.start, 0.8,
                                       color=colors[i % len(colors)], alpha=0.9))
    end = max(pm.get_end_time(), 1)
    pitches = [n.pitch for inst in pm.instruments for n in inst.notes] or [60]
    ax.set_xlim(0, end)
    ax.set_ylim(min(pitches) - 2, max(pitches) + 2)
    ax.set_title(title, color="white", fontsize=14)
    ax.tick_params(colors="#888")
    ax.set_xlabel("seconds", color="#888")
    ax.set_ylabel("pitch", color="#888")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def fail_card(png_path, title, reason):
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="#0b0b12")
    ax.axis("off")
    ax.text(0.5, 0.6, "NO VALID MUSIC", color="#ef476f", fontsize=40, ha="center", weight="bold")
    ax.text(0.5, 0.3, f"{title}\nreason: {reason}", color="#aaa", fontsize=12, ha="center")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    results = json.load(open(f"results/{args.tag}.json", encoding="utf-8"))
    out = Path("media") / args.tag
    out.mkdir(parents=True, exist_ok=True)
    for r in results:
        s = r["samples"][args.sample]
        stem = out / r["src_id"]
        title = f"{args.tag.upper()} | {r['tag']} | score {s['total']:.2f}"
        abc = s.get("abc")
        if not abc:
            fail_card(f"{stem}.png", title, s.get("fail", "?"))
            continue
        Path(f"{stem}.abc").write_text(abc, encoding="utf-8")
        sp.run([ABC2MIDI, f"{stem}.abc", "-o", f"{stem}.mid"], capture_output=True, timeout=60)
        if not Path(f"{stem}.mid").exists() or s["total"] == 0:
            fail_card(f"{stem}.png", title, s.get("fail", "abc2midi failed"))
            continue
        sp.run(["fluidsynth", "-ni", "-g", "0.8", "-F", f"{stem}.wav", "-r", "44100", SF2, f"{stem}.mid"],
               capture_output=True, timeout=300)
        piano_roll(f"{stem}.mid", f"{stem}.png", title)
    print(f"rendered {len(results)} pieces -> {out}")


if __name__ == "__main__":
    main()
