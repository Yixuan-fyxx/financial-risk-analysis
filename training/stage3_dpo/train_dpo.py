"""Stage 3 (RL pipeline): DPO on the Stage-2 merged model, using preference
pairs from `training/data_gen/build_preference_dataset.py` (rejection
sampling scored by `report_verifier` in place of human/LLM preference
labels).

DPO instead of PPO: no separate reward-model network to train, one policy
model to hold in memory, single-GPU-friendly, and the reference policy is
obtained for free by disabling the LoRA adapter (no second copy of the model
needed) — see the `peft_config` + `ref_model=None` combination below, which
is the standard way trl's DPOTrainer supports this. See training/README.md
for why PPO was left as an optional extension rather than a requirement.

Requires training/requirements-train.txt and a GPU. Usage:
    python -m training.stage3_dpo.train_dpo --config training/configs/dpo.yaml
"""

from __future__ import annotations

import argparse

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from training.common.chat_format import render_prompt
from training.common.config import load_yaml
from training.common.merge_utils import merge_lora_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="training/configs/dpo.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"], torch_dtype="auto")

    dataset = load_dataset("json", data_files=cfg["dataset_path"], split="train")

    def to_dpo_fields(example: dict) -> dict:
        prompt = render_prompt(example["system_prompt"], example["user_prompt"])
        return {
            "prompt": prompt,
            "chosen": example["chosen"] + tokenizer.eos_token,
            "rejected": example["rejected"] + tokenizer.eos_token,
        }

    dataset = dataset.map(to_dpo_fields, remove_columns=dataset.column_names)

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_cfg = cfg["train"]
    dpo_config = DPOConfig(
        output_dir=cfg["output_dir"],
        beta=train_cfg["beta"],
        max_length=cfg["max_seq_length"],
        max_prompt_length=cfg["max_prompt_length"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        bf16=train_cfg["bf16"],
        seed=train_cfg["seed"],
        report_to=[],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        precompute_ref_log_probs=True,  # cache the adapter-disabled reference pass once, up front,
        # instead of re-running it interleaved with the policy's backward pass every step
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # LoRA + peft_config: reference policy = adapter disabled, no 2nd model copy
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"Stage 3 DPO LoRA adapter saved to {cfg['output_dir']}")

    merged_dir = f"{cfg['output_dir']}_merged"
    merge_lora_adapter(cfg["model_name_or_path"], cfg["output_dir"], merged_dir)


if __name__ == "__main__":
    main()
