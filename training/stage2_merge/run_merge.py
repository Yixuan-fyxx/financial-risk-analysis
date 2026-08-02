"""Stage 2: fold the Stage-1 LoRA adapter into full weights, then merge that
SFT model back with the untouched base model via mergekit — the "SFT can
overfit/forget general capability, merge back some base weight to
compensate" step.

Two sub-steps, both driven by `training/configs/merge.yaml`:
  1. `peft` `merge_and_unload()`: LoRA adapter (`sft_model_dir`) + base ->
     one full-weight "pure SFT" checkpoint (`sft_merged_dir`).
  2. `mergekit-yaml` (subprocess): blends `sft_merged_dir` with the
     untouched base model per `mergekit_config` -> `output_dir`.

Requires training/requirements-train.txt and a GPU — not run as part of the
CPU test suite. Run `training/stage2_merge/eval_general_capability.py`
afterwards to quantify what the merge traded off.

Usage:
    python -m training.stage2_merge.run_merge --config training/configs/merge.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from training.common.config import load_yaml
from training.common.merge_utils import merge_lora_adapter

# Workaround for https://github.com/arcee-ai/mergekit/issues/681 (open, unfixed as of
# this writing): several of mergekit's pydantic models reference `torch` in forward-ref
# type hints that fail to resolve on first use ("`ConfiguredModuleArchitecture`/
# `ConfiguredArchitectureInfo` is not fully defined"), regardless of the pydantic 2.10.x
# patch version. We can't patch this from our own process because `mergekit-yaml` runs
# as a separate subprocess, so instead of invoking that console script we run an inline
# snippet that imports torch and force-rebuilds every mergekit pydantic model *before*
# mergekit's own code tries to instantiate one for the first time (which is what
# triggers the crash), then calls the same entry point the console script does.
_MERGEKIT_PATCH = """
import importlib
import sys

import pydantic
import torch

def _rebuild(mod_name):
    try:
        mod = importlib.import_module(mod_name)
    except ImportError:
        return
    for name in dir(mod):
        obj = getattr(mod, name, None)
        if isinstance(obj, type) and issubclass(obj, pydantic.BaseModel):
            try:
                obj.model_rebuild(force=True, _types_namespace={"torch": torch})
            except Exception:
                pass

for _mod_name in ("mergekit.architecture", "mergekit.plan", "mergekit.merge_methods.base", "mergekit.config"):
    _rebuild(_mod_name)

config_path, output_dir = sys.argv[1], sys.argv[2]
sys.argv = ["mergekit-yaml", config_path, output_dir]
from mergekit.scripts.run_yaml import main
main()
"""


def run_mergekit(mergekit_config: dict, output_dir: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
        yaml.safe_dump(mergekit_config, fh, allow_unicode=True)
        config_path = fh.name
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-c", _MERGEKIT_PATCH, config_path, output_dir], check=True)
    print(f"mergekit ({mergekit_config['merge_method']}) output -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="training/configs/merge.yaml")
    parser.add_argument("--sft-base-model", default="Qwen/Qwen2.5-1.5B", help="Must match configs/sft.yaml.")
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    merge_lora_adapter(args.sft_base_model, cfg["sft_model_dir"], cfg["sft_merged_dir"])
    run_mergekit(cfg["mergekit_config"], cfg["output_dir"])


if __name__ == "__main__":
    main()
