# mimo-music-grpo

Teach an open model to write music with **GRPO**, using the **music RL environment Xiaomi open-sourced
with MiMo-V2.6**: their prompts and their rule-based grader. One rented GPU, about 1 h 40 min of training, about $12.

| | Base Qwen3-8B | After 30 GRPO steps |
|---|---|---|
| Valid, playable music (Xiaomi's strict grader, 40 unseen prompts × 4) | 1/160 (0.6%) | **19/160 (11.9%)** |
| Prompts with ≥1 valid piece | 1/40 | 13/40 |
| Avg answer length | 2171 tokens | 1256 tokens |

**Trained model:** [Prompt48/Qwen3-8B-MiMo-Music-GRPO](https://huggingface.co/Prompt48/Qwen3-8B-MiMo-Music-GRPO) (merged + LoRA adapter)
· **GGUF:** [Prompt48/Qwen3-8B-MiMo-Music-GRPO-GGUF](https://huggingface.co/Prompt48/Qwen3-8B-MiMo-Music-GRPO-GGUF) (use **Q8_0**)

Video walkthrough: Prompt Engineer 48 on YouTube.

## What comes from Xiaomi
| Piece | Source |
|---|---|
| 1,000 composition prompts (496 English) | [`XiaomiMiMo/MiMo-V2.6-RL-oss`](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss) → `music.parquet` |
| Grader (18 features, 6 groups, "how human-like") | `mimo_scorer/` = unchanged copy of [`XiaomiMiMo/verl` · `recipes/design/music/scorer`](https://github.com/XiaomiMiMo/verl/tree/mimo-oss/recipes/design/music) (Apache-2.0) |
| Group-relative length penalty | MiMo-V2.6 report Eq. 4, ported in `reward.py` |
| GRPO, no KL, clip 0.2 | same as their recipe configs |

The trainer is [TRL](https://github.com/huggingface/trl) + vLLM (not Xiaomi's 64-GPU verl setup), so it fits one GPU.

## Run it (1× H100 80GB, ~2.5 h)
```bash
bash setup_pod.sh     # abc2midi, fluidsynth, TRL, vLLM; downloads prompts; runs precheck
bash run_all.sh       # before-eval -> train -> after-eval -> audio, charts, blind test
```

| File | What it does |
|---|---|
| `prep_data.py` | Train/test split of Xiaomi's English prompts (40 held out, never trained on) |
| `precheck.py` | Proves the grader is live before you spend GPU money (without `abc2midi` every score is silently 0) |
| `reward.py` | Xiaomi's grader + training-only blank-line credit + Eq. 4 length penalty |
| `train_grpo.py` | GRPO: Qwen3-8B (thinking off), LoRA r=64, 8 prompts × 8 answers/step, 30 steps, lr 2e-5 |
| `generate.py` | Before/after answers on the 40 held-out prompts, same seed |
| `render.py`, `charts.py`, `blind_fair.py` | Audio + piano rolls, charts, fair blind A/B test |
| `merge_and_gguf.sh` | Merge LoRA → Qwen3-8B, build GGUFs, push to HF |
| `gguf_eval.py`, `quant_check.sh` | Score GGUF quants on the same test set |

## The key finding: blank lines
No base model we tried could pass Xiaomi's grader (Qwen2.5-3B/7B 0%, MiMo-Distill-9B 0% at 8K tokens, Qwen3-8B 1–6%).
In ABC notation **a blank line ends the tune**, and 74 of 80 Qwen3-8B failures had blank lines between sections;
deleting them alone gave 16% valid. So during **training only**, a piece that fails *only* because of blank lines
is graded with them removed and gets 30% credit. All reported numbers use Xiaomi's strict grader, unmodified.

That matches the MiMo-V2.6 report: RL can only improve what the model can already do sometimes.

## Quantization check (same 40 prompts × 4)
| Version | Valid music |
|---|---|
| Merged bf16 (vLLM) | 14.4% |
| **Q8_0 GGUF** | **18.8%** |
| Q4_K_M GGUF | 5.0% |

Q4_K_M loses most of the RL gain. Use Q8_0. Prompt with long, detailed briefs like the training data.

## Proof
`runs/train_run/` has the real logs (setup, probes, the out-of-memory crash, training), `trainer_state.json`,
charts, and every before/after answer with its score. `runs/quant_check/` has the quant results and a
Q8_0-generated waltz (ABC, MIDI, MP3; Xiaomi score 0.507).

## Limitations
About 88% of answers still fail the strict grader. The grader measures statistical human-likeness, not beauty. This is a small replica
of Xiaomi's method, not a Xiaomi model.

## License
Apache-2.0. `mimo_scorer/` is © Xiaomi / Bytedance (Apache-2.0, headers kept). Base model: Qwen3-8B (Apache-2.0).
