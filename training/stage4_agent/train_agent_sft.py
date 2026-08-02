"""Stage 4: LoRA SFT for tool-calling, on top of the Stage-3 DPO model, using
ReAct trajectories from `training/data_gen/build_agent_trajectories.py`.

This is the same SFT recipe as Stage 1 (`training.common.lora_sft`) applied
to a different dataset/base checkpoint — Stage 1 taught report-writing
*format*, this stage teaches *when to call which tool with what arguments*.

Merges the resulting adapter into full weights afterwards
(`{output_dir}_merged`) — that directory is the final pipeline artifact and
what `agent_runtime.py` loads.

Requires training/requirements-train.txt and a GPU. Usage:
    python -m training.stage4_agent.train_agent_sft --config training/configs/agent_sft.yaml
"""

from __future__ import annotations

import argparse

from training.common.config import load_yaml
from training.common.lora_sft import run_lora_sft
from training.common.merge_utils import merge_lora_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="training/configs/agent_sft.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    run_lora_sft(cfg)

    merged_dir = f"{cfg['output_dir']}_merged"
    merge_lora_adapter(cfg["model_name_or_path"], cfg["output_dir"], merged_dir)
    print(f"Final pipeline model: {merged_dir}")


if __name__ == "__main__":
    main()
