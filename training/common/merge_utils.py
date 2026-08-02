"""Folds a LoRA adapter into full model weights via `peft.PeftModel.merge_and_unload`.

Every stage after Stage 1 loads its base model with `AutoModelForCausalLM
.from_pretrained`, which needs a full-weights directory — it cannot load a
LoRA-adapter-only directory as a base model. So each LoRA stage (SFT, DPO,
agent SFT) is followed by a merge-into-full-weights step before the next
stage (or mergekit, or inference) can build on top of it. This is the one
place that logic lives, reused by `stage2_merge/run_merge.py`,
`stage3_dpo/train_dpo.py`, and `stage4_agent/train_agent_sft.py`.
"""

from __future__ import annotations

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.common.config import patch_tokenizer_extra_special_tokens_list_bug


def merge_lora_adapter(base_model_path: str, adapter_dir: str, out_dir: str) -> None:
    patch_tokenizer_extra_special_tokens_list_bug()
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Merged LoRA adapter ({adapter_dir}) into full weights at {out_dir}")
