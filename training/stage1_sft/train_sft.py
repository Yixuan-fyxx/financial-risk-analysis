"""Stage 1: LoRA SFT on the pretrained base model, teaching it to produce
v4-structured risk reports from `training/datasets/sft.jsonl`
(see `training/data_gen/build_sft_dataset.py`).

Requires training/requirements-train.txt (transformers/peft/trl/accelerate)
and a GPU — not run as part of the CPU test suite.

Usage:
    python -m training.stage1_sft.train_sft --config training/configs/sft.yaml
"""

from __future__ import annotations

import argparse

from training.common.config import load_yaml
from training.common.lora_sft import run_lora_sft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="training/configs/sft.yaml")
    args = parser.parse_args()
    run_lora_sft(load_yaml(args.config))


if __name__ == "__main__":
    main()
