"""Shared LoRA-SFT training loop used by both Stage 1 (report-writing SFT)
and Stage 4 (agent tool-use SFT) — the two stages are the same recipe
(next-token SFT on chat-formatted `messages`) over different datasets/base
checkpoints, so the loop itself isn't duplicated between them.
"""

from __future__ import annotations

import inspect

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from training.common.chat_format import render_messages
from training.common.config import patch_tokenizer_extra_special_tokens_list_bug


def run_lora_sft(cfg: dict) -> None:
    patch_tokenizer_extra_special_tokens_list_bug()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"], torch_dtype="auto")

    dataset = load_dataset("json", data_files=cfg["dataset_path"], split="train")

    def to_text(example: dict) -> dict:
        return {"text": render_messages(example["messages"], eos_token=tokenizer.eos_token)}

    dataset = dataset.map(to_text, remove_columns=dataset.column_names)

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
    # trl renamed SFTConfig's sequence-length arg (max_seq_length -> max_length) in
    # newer releases; requirements-train.txt pins trl>=0.11 with no upper bound, so
    # detect whichever name the installed version accepts instead of hardcoding one.
    max_len_kwarg = (
        "max_length" if "max_length" in inspect.signature(SFTConfig.__init__).parameters else "max_seq_length"
    )
    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        dataset_text_field="text",
        packing=False,
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
        **{max_len_kwarg: cfg["max_seq_length"]},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"LoRA adapter saved to {cfg['output_dir']}")
