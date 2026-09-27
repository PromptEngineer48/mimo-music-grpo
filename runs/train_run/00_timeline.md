# Video log — MiMo music GRPO run (2026-09-27)

Everything below is real. Raw logs are in this folder.

## Setup
- Rented via RunPod REST API: 1x NVIDIA H100 80GB HBM3, Secure Cloud, $3.49/hr, 120GB disk.
- Pod 1: SSH failed ("Permission denied (publickey)") — laptop key has a passphrase. Deleted; pod 2 used a passphrase-free key.
- Stack: TRL 1.14.0, vLLM 0.30.0, torch 2.13 (cu130), transformers 5.17 (versions.txt).
- Precheck (01_setup.log): hand-written test waltz scored 0.619 by Xiaomi's grader
  (rhythm 11.8, texture 73.1, acoustic 75, tonal 81.8, register 100, structure 91.7); plain text scored 0.0.

## Finding 1 — base models can't pass Xiaomi's music grader (cold-open beat)
40 held-out English prompts, strict Xiaomi grader:
| Model | Valid playable music | Notes |
|---|---|---|
| Qwen2.5-3B-Instruct | 0/160 (0%) | "M:203/4", "Eb4 Eb4 Eb4..." gibberish (before_qwen2.5-3b_0pct_valid.json) |
| Qwen2.5-7B-Instruct | 0/80 | abc2midi syntax errors |
| MiMo-V2.6-Distill-Qwen-9B (Xiaomi's), 8K budget | 0/80 | Thinks so long (avg 8192 tokens) it never writes the music. Xiaomi's recipe allows 100K. |
| MiMo-V2.6-Distill-Qwen-9B, 16K budget | 1/40 (2.5%) | |
| Qwen3-8B, thinking on | 5/80 (6.2%) | ~6.5K tokens per answer |
| Qwen3-8B, thinking off | 1/80 (1.2%) | ~1.4K tokens |
Chinese short (32-56 bar) prompts: no better (Qwen3-8B 1.2% / 5%).

## Finding 2 — the real bug: blank lines
- In ABC notation a blank line ENDS the tune; Xiaomi's grader rejects any piece containing one.
- 74 of 80 Qwen3-8B rejects had blank lines between sections.
- Deleting blank lines alone: 1% -> 16% valid (13/80).
- Matches the paper: RL needs a model that already sometimes succeeds (Xiaomi SFTs on 77B tokens first).

## Training reward design (disclose on video)
- Reward = Xiaomi's score. If a piece fails ONLY because of blank lines, it is graded with them
  removed but gets 30% credit -> model is paid to learn the "no blank lines" rule.
- Plus Xiaomi's group-relative length penalty (paper Eq. 4).
- FINAL numbers use Xiaomi's strict grader only, on 40 prompts never trained on.

## Training attempts
1. TRL 1.14 removed `max_prompt_length` -> crash. Then CUDA OOM (18.5GB logits alloc for 8 x 5K-token answers) (06a_train_OOM.log). Fix: 2 answers per micro-batch.
2. 16 prompts x 8 answers: step-1 reward 0.013, 75% of groups had signal, but 7.2 min/step (06b_train_16prompts_step1.log). Too slow.
3. Final: Qwen3-8B (thinking off) + LoRA r=64, 8 prompts x 8 answers/step, lr 2e-5, 30 steps, no KL, clip 0.2, ~3.3 min/step (06_train.log).

## RESULTS (strict Xiaomi grader, 40 held-out prompts x 4 samples, seed 48)
| | Before (Qwen3-8B) | After 30 GRPO steps |
|---|---|---|
| Valid playable music | 1/160 (0.6%) | 19/160 (11.9%) — ~20x |
| Prompts with >=1 valid piece | 1/40 | 13/40 |
| Mean Xiaomi score (all) | 0.002 | 0.041 |
| Avg answer length | 2171 tokens | 1256 tokens (-42%, length penalty) |
| Pieces with blank lines | 148/160 | 132/160 (habit only partly unlearned) |
- Training reward per step: 0.013 -> ~0.10-0.15 by steps 27-30 (charts/4_reward_curve.png)
- Group scores of valid pieces after: rhythm 24, texture 61, acoustic 47, tonal 41, register 14, structure 52
  (before has only ONE valid piece, so its radar is a single sample — don't over-claim the radar)
- Blind test: 13 pairs, blind/index.html. AFTER = first valid sample per prompt (disclose),
  BEFORE = base sample with blank lines removed (favour to base). Answer key blind/ANSWER_KEY.json.

## Honest caveats for the video
- 12% valid is still low; 30 steps (~1.7h) is tiny vs Xiaomi's run.
- Training reward had one shaping step. Final numbers don't use it.
- Model = Qwen3-8B + LoRA, not Xiaomi's 9B (it thinks too long for our token budget).

## Cost / time
- Pod 1 ~20 min + pod 2 ~3h15m at $3.49/hr ≈ $12-13 total. Both pods deleted.
