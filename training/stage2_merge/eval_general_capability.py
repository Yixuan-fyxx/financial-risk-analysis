"""Quantifies what Stage 2's merge traded off: compares the untouched base
model, the pure Stage-1 SFT model, and the merged model on
  - general capability (loss on generic non-finance reference continuations)
  - domain quality (report_verifier score on the 3 real companies)

Expected story if the merge worked as intended: SFT has the best domain
score but a higher (worse) general-capability loss than base; the merged
model's general-capability loss moves back toward base while domain score
stays meaningfully above base. Print/save this table — it's the headline
number for the "model merging" part of the pipeline.

Requires training/requirements-train.txt and a GPU. Usage:
    python -m training.stage2_merge.eval_general_capability \
        --base-model Qwen/Qwen2.5-1.5B \
        --sft-merged-model training/outputs/stage1_sft_merged \
        --merged-model training/outputs/stage2_merged \
        --out training/outputs/stage2_merge_eval.json
"""

from __future__ import annotations

import argparse
import json

from transformers import AutoModelForCausalLM, AutoTokenizer

from training.data_gen.build_preference_dataset import build_prompts
from training.eval.scoring import domain_verifier_scores, general_capability_loss, summarize


def evaluate_checkpoint(label: str, model_path: str, domain_prompts: list[dict]) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    model.eval()

    general_loss = general_capability_loss(model, tokenizer)
    domain_scores = domain_verifier_scores(model, tokenizer, domain_prompts)
    return summarize(label, general_loss, domain_scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--sft-merged-model", default="training/outputs/stage1_sft_merged")
    parser.add_argument("--merged-model", default="training/outputs/stage2_merged")
    parser.add_argument("--out", default="training/outputs/stage2_merge_eval.json")
    args = parser.parse_args()

    domain_prompts = build_prompts()

    results = [
        evaluate_checkpoint("base", args.base_model, domain_prompts),
        evaluate_checkpoint("stage1_sft_merged", args.sft_merged_model, domain_prompts),
        evaluate_checkpoint("stage2_merged", args.merged_model, domain_prompts),
    ]

    print(f"{'checkpoint':<20}{'general_loss':<16}{'domain_score':<16}")
    for r in results:
        print(f"{r['checkpoint']:<20}{r['general_capability_loss']:<16}{r['domain_verifier_score_mean']:<16}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
