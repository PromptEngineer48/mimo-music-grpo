"""Rewards for the music demo.

1. music_reward       -- Xiaomi's own scorer (mimo_scorer, copied verbatim from
                         XiaomiMiMo/verl recipes/design/music/scorer). 0..1,
                         "how human-like is this music". Invalid ABC -> 0.
2. length_penalty     -- port of MiMo-V2.6 Eq. (4) / verl/utils/length_penalty.py:
                         among PASSING rollouts of the same prompt, penalize ones
                         much longer than the group's reference length.

TRL calls reward functions on a flat list where each prompt appears
num_generations times; we regroup by prompt text to get GRPO groups.
"""

from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from mimo_scorer import do, extract_abc

_POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("REWARD_WORKERS", "32")))

# Eq. (4) hyper-parameters (X, delta, s, gamma, pass gate A, percentile B)
LP_MAX = float(os.environ.get("LP_MAX", "0.1"))       # X: max deduction
LP_DELTA = float(os.environ.get("LP_DELTA", "0.5"))   # tolerated excess (150% of ref)
LP_SAT = float(os.environ.get("LP_SAT", "1.5"))       # excess where penalty saturates
LP_GAMMA = float(os.environ.get("LP_GAMMA", "2.0"))   # ramp exponent
LP_GATE = float(os.environ.get("LP_GATE", "0.5"))     # A: min group pass rate
LP_PCT = float(os.environ.get("LP_PCT", "50"))        # B: percentile
PASS_THRESHOLD = float(os.environ.get("PASS_THRESHOLD", "0.01"))  # score>0 == valid music


def _text(completion) -> str:
    if isinstance(completion, list):  # conversational format
        return completion[-1]["content"]
    return completion


def score_one(text: str) -> dict:
    """Full breakdown: total (0..1), per-group scores, and why it failed if it did."""
    abc = extract_abc(text)
    if not abc:
        return {"total": 0.0, "fail": "no_abc"}
    try:
        r = do({"key": "demo", "id": 0, "rep": 0, "abc": abc, "abc_len": len(abc), "nvoice": 0})
    except Exception as e:  # Abc2MidiMissing must be loud
        raise
    if r.get("skip"):
        return {"total": 0.0, "fail": r["skip"], "abc": abc}
    if r.get("reject"):
        return {"total": 0.0, "fail": "reject(err/bar/blank/ch_conflict)", "abc": abc, **_extra(r)}
    if "total" not in r:
        return {"total": 0.0, "fail": r.get("scorer_skip", "no_total"), "abc": abc}
    return {"total": max(0.0, min(1.0, r["total"] / 100.0)), "groups": r["groups"], "abc": abc, **_extra(r)}


def _extra(r):
    return {k: r.get(k) for k in ("n_chan", "pit_range", "err", "bar")}


def music_reward(prompts, completions, **kwargs):
    """Pure Xiaomi grader (strict). Used for evaluation."""
    texts = [_text(c) for c in completions]
    return [s["total"] for s in _POOL.map(score_one, texts)]


BLANK_CREDIT = float(os.environ.get("BLANK_CREDIT", "0.3"))


def _train_score(text: str) -> float:
    """Xiaomi grader + one shaping step so GRPO gets a signal from a weak base model.

    In ABC a blank line ENDS the tune, so Xiaomi's grader rejects any piece with one.
    Base models do this ~80% of the time. If the piece is otherwise valid once blank
    lines are removed, it earns BLANK_CREDIT x its Xiaomi score (so a clean piece
    always beats the same piece with blank lines). Evaluation never uses this.
    """
    s = score_one(text)
    if s["total"] > 0 or not s.get("abc"):
        return s["total"]
    stripped = "\n".join(l for l in s["abc"].split("\n") if l.strip())
    if stripped == s["abc"]:
        return 0.0
    return BLANK_CREDIT * score_one("```abc\n" + stripped + "\n```")["total"]


def music_reward_train(prompts, completions, **kwargs):
    texts = [_text(c) for c in completions]
    return list(_POOL.map(_train_score, texts))


def _group_ids(prompts):
    keys = [str(p) for p in prompts]
    ids, seen = [], {}
    for k in keys:
        ids.append(seen.setdefault(k, len(seen)))
    return ids


def length_penalty(prompts, completions, completion_ids=None, **kwargs):
    """Returns a NEGATIVE adjustment (0 or below). Add it to music_reward."""
    lengths = [len(c) for c in completion_ids] if completion_ids is not None else [len(_text(c)) for c in completions]
    base = music_reward_train(prompts, completions)
    groups = defaultdict(list)
    for i, g in enumerate(_group_ids(prompts)):
        groups[g].append(i)
    out = [0.0] * len(completions)
    for idx in groups.values():
        passed = [i for i in idx if base[i] > PASS_THRESHOLD]
        if not passed or len(passed) / len(idx) <= LP_GATE:
            continue  # hard prompt: leave room to explore
        ref = float(np.percentile([lengths[i] for i in passed], LP_PCT))
        for i in passed:
            excess = lengths[i] / max(ref, 1) - 1
            t = np.clip((excess - LP_DELTA) / (LP_SAT - LP_DELTA), 0, 1)
            out[i] = -LP_MAX * float(t) ** LP_GAMMA
    return out
