"""GRPO on Xiaomi's music environment, 1 GPU, TRL + vLLM.

    python train_grpo.py --model Qwen/Qwen2.5-3B-Instruct --max_steps 120

MiMo-V2.6 ideas used:
  - Xiaomi's rule-based music verifier as reward        (paper 7.2, recipes/design/music)
  - group-relative length penalty on passing rollouts   (Eq. 4)
  - no KL term (beta=0), clip 0.2, lr 1e-6-ish          (recipes/*/config)
  - groups with identical rewards give zero advantage   (dynamic-sampling effect)
"""

import argparse
import json

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from peft import LoraConfig

from reward import length_penalty, music_reward_train


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="ckpt/music-grpo")
    ap.add_argument("--max_steps", type=int, default=120)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--prompts_per_step", type=int, default=8)
    ap.add_argument("--max_completion", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--no_length_penalty", action="store_true")
    ap.add_argument("--report_to", default="none")
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--micro_bs", type=int, default=2)
    args = ap.parse_args()

    train = Dataset.from_list([{"prompt": r["prompt"]} for r in load("data/train.jsonl")])
    gen_batch = args.prompts_per_step * args.num_generations

    cfg = GRPOConfig(
        output_dir=args.out,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        warmup_steps=0,
        per_device_train_batch_size=args.micro_bs,
        gradient_accumulation_steps=gen_batch // args.micro_bs,
        num_generations=args.num_generations,
        vllm_max_model_length=6144,
        max_completion_length=args.max_completion,
        temperature=1.0,
        beta=0.0,
        epsilon=0.2,
        loss_type="dapo",
        mask_truncated_completions=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=1,
        save_steps=10,
        save_total_limit=3,
        report_to=args.report_to,
        run_name="mimo-music-grpo",
        chat_template_kwargs={"enable_thinking": args.think},
    )
    peft = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.0,
                      target_modules="all-linear", task_type="CAUSAL_LM")

    rewards = [music_reward_train] if args.no_length_penalty else [music_reward_train, length_penalty]
    trainer = GRPOTrainer(model=args.model, reward_funcs=rewards, args=cfg, train_dataset=train, peft_config=peft)
    trainer.train()
    trainer.save_model(args.out + "/adapter")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(args.out + "/final")
    trainer.processing_class.save_pretrained(args.out + "/final")


if __name__ == "__main__":
    main()
