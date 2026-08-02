"""Final evaluation: runs base / stage1_sft / stage2_merged / stage3_dpo /
stage4_agent through the same held-out prompts and produces one comparison
table — the headline numbers for the whole pipeline, not just Stage 2's merge.

Two families of metrics, reusing `training/eval/scoring.py`:
  - general_capability_loss / domain_verifier_score_mean: computed for every
    checkpoint, so the base->SFT->merge->DPO trajectory of "domain skill
    gained vs. general capability retained" is visible end to end.
  - agent task_completion_rate / agent_domain_score_mean: only meaningful
    for stage4_agent (the only checkpoint trained to emit Action/Observation
    trajectories) — computed by actually running `agent_runtime.run_agent`
    against the real fin_risk tools, not just scoring a single generation.

Held-out prompts = the 3 real companies, via `build_preference_dataset
.build_prompts` — pass `--as-of-date` different from the SFT dataset's
default (`training/data_gen/build_sft_dataset.py`'s `DEFAULT_AS_OF_DATE`) so
retrieval recency-weighting isn't running on the exact training snapshot.
Evaluating generalization to genuinely unseen financial figures (synthetic
variants with a seed the SFT/DPO/agent datasets never used) is a natural
extension noted in training/README.md but not implemented here — the 3 real
companies are the only ground-truth-labeled cases available.

Requires training/requirements-train.txt and a GPU. Usage:
    python -m training.eval.run_eval --out training/outputs/final_eval.json
"""

from __future__ import annotations

import argparse
import json

from training.data_gen.build_preference_dataset import DEFAULT_AS_OF_DATE, build_prompts
from training.eval.scoring import domain_verifier_scores, general_capability_loss, summarize
from training.verifier.report_verifier import verify_report

CHECKPOINTS = [
    ("base", "Qwen/Qwen2.5-1.5B"),
    ("stage1_sft", "training/outputs/stage1_sft_merged"),
    ("stage2_merged", "training/outputs/stage2_merged"),
    ("stage3_dpo", "training/outputs/stage3_dpo_merged"),
    ("stage4_agent", "training/outputs/stage4_agent_merged"),
]

HELD_OUT_COMPANY_IDS = ["000333", "600585", "3333HK"]


def agent_trace_outcome(trace: str, evidence_count: int) -> dict:
    """Pure post-processing of an agent trace: did it reach a Final report
    (as opposed to hitting max_turns), and if so, how does that report score?
    Kept separate from model/generation code so it's unit-testable on CPU.
    """
    completed = "Final:" in trace and "达到最大轮数" not in trace
    score = None
    if completed:
        report_text = trace.split("Final:", 1)[1].strip()
        score = verify_report(report_text, evidence_count=evidence_count, data_coverage=1.0).score
    return {"completed": completed, "score": score}


def evaluate_agent_capability(model_path: str, max_turns: int = 4) -> dict:
    from fin_risk.data.loader import load_company
    from training.stage4_agent.agent_runtime import _build_hf_generate_fn, run_agent

    generate_fn = _build_hf_generate_fn(model_path)
    outcomes = []
    for company_id in HELD_OUT_COMPANY_IDS:
        company = load_company(company_id)
        user_prompt = f"请分析{company.name}({company.ticker})的风险状况,写一份风险报告。"
        trace = run_agent(generate_fn, user_prompt, max_turns=max_turns)
        outcomes.append(agent_trace_outcome(trace, evidence_count=4))

    completed_scores = [o["score"] for o in outcomes if o["completed"]]
    return {
        "task_completion_rate": round(sum(o["completed"] for o in outcomes) / len(outcomes), 4),
        "agent_domain_score_mean": (
            round(sum(completed_scores) / len(completed_scores), 4) if completed_scores else None
        ),
    }


def format_table(results: list[dict]) -> str:
    headers = [
        "checkpoint", "general_capability_loss", "domain_verifier_score_mean",
        "task_completion_rate", "agent_domain_score_mean",
    ]
    lines = ["".join(f"{h:<28}" for h in headers)]
    for r in results:
        lines.append("".join(f"{str(r.get(h, '')):<28}" for h in headers))
    return "\n".join(lines)


def evaluate_checkpoint(label: str, model_path: str, domain_prompts: list[dict], run_agent_eval: bool) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from training.common.config import patch_tokenizer_extra_special_tokens_list_bug

    patch_tokenizer_extra_special_tokens_list_bug()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    model.eval()

    general_loss = general_capability_loss(model, tokenizer)
    domain_scores = domain_verifier_scores(model, tokenizer, domain_prompts)
    result = summarize(label, general_loss, domain_scores)

    if run_agent_eval:
        result.update(evaluate_agent_capability(model_path))
    else:
        result["task_completion_rate"] = None
        result["agent_domain_score_mean"] = None

    del model  # free GPU memory before the next checkpoint loads
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of-date", default=DEFAULT_AS_OF_DATE)
    parser.add_argument("--out", default="training/outputs/final_eval.json")
    args = parser.parse_args()

    domain_prompts = build_prompts(as_of_date=args.as_of_date)

    results = [
        evaluate_checkpoint(label, path, domain_prompts, run_agent_eval=(label == "stage4_agent"))
        for label, path in CHECKPOINTS
    ]

    print(format_table(results))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
