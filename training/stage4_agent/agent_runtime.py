"""Runs the Stage-4 agent model end-to-end: the model emits an Action, this
script executes the *real* fin_risk tool via `agent.tools.call_tool`, feeds
back a real Observation, and loops until the model emits a Final report.

This is the concrete proof that the pipeline produced more than a
checkpoint: the trained model actually drives fin_risk's risk_scoring/rag
modules through multi-step tool use — the same modules `ReportPipeline`
calls internally, but now orchestrated by a learned policy instead of the
hardcoded `_build_rag_query` flow in pipeline.py.

Generation is stopped (via a custom `StoppingCriteria`) the instant the
model starts writing "\\nObservation:" itself — Stage 4 trained it to always
follow an Action with an Observation, so at inference time we must cut it
off right there and splice in the *real* tool result instead of letting it
hallucinate one. If no such attempt appears before EOS/max_new_tokens, the
model wasn't trying to fake an Observation — it either finished with a
Final report or ran out of budget; either way that turn is terminal.

Requires training/requirements-train.txt and a GPU (or CPU, just slow).

Usage:
    python -m training.stage4_agent.agent_runtime \
        --model-path training/outputs/stage4_agent_merged --company-id 600585
"""

from __future__ import annotations

import argparse
import json
from typing import Callable, Optional

from fin_risk.agent.tools import ToolError, call_tool
from fin_risk.data.loader import load_company
from training.common.chat_format import render_prompt
from training.data_gen.build_agent_trajectories import SYSTEM_PROMPT

STOP_MARKER = "\nObservation:"

# context-so-far -> newly generated continuation. Injected so `run_agent`'s
# turn-taking/tool-dispatch logic is unit-testable without transformers/torch
# (see build_preference_dataset.py's `GenerateFn` for the same pattern).
GenerateFn = Callable[[str], str]


def _extract_action(text: str) -> Optional[dict]:
    for line in text.splitlines():
        if line.startswith("Action: "):
            try:
                return json.loads(line[len("Action: "):])
            except json.JSONDecodeError:
                return None
    return None


def run_agent(generate_fn: GenerateFn, user_prompt: str, max_turns: int = 4) -> str:
    """Returns the full Thought/Action/Observation/.../Final trace."""
    trace = ""
    context = render_prompt(SYSTEM_PROMPT, user_prompt)

    for _ in range(max_turns):
        generated = generate_fn(context)

        if STOP_MARKER not in generated:
            # No attempted Observation: either a completed Final report (EOS
            # hit) or we ran out of max_new_tokens mid-generation. Terminal
            # either way — nothing left to feed a real Observation into.
            return trace + generated

        action_text = generated.split(STOP_MARKER)[0]
        trace += action_text
        context += action_text

        action = _extract_action(action_text)
        if action is None:
            observation = {"error": "无法解析 Action,请输出合法 JSON,如 {\"tool\": ..., \"arguments\": {...}}"}
        else:
            try:
                observation = call_tool(action["tool"], action.get("arguments", {}))
            except ToolError as exc:
                observation = {"error": str(exc)}

        observation_line = f"\nObservation: {json.dumps(observation, ensure_ascii=False)}\n"
        trace += observation_line
        context += observation_line

    return trace + "\n[达到最大轮数,未能生成最终报告]"


def _build_hf_generate_fn(model_path: str, max_new_tokens: int = 1200) -> GenerateFn:
    """Lazily imports transformers/torch so this module stays importable/
    testable without the training extras installed. Only called from `main()`."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

    from training.common.config import patch_tokenizer_extra_special_tokens_list_bug

    patch_tokenizer_extra_special_tokens_list_bug()

    class StopOnSubstring(StoppingCriteria):
        def __init__(self, tokenizer, prompt_len: int, stop_string: str):
            self.tokenizer = tokenizer
            self.prompt_len = prompt_len
            self.stop_string = stop_string

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            text = self.tokenizer.decode(input_ids[0][self.prompt_len:], skip_special_tokens=True)
            return self.stop_string in text

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    model.eval()

    def generate_fn(context: str) -> str:
        input_ids = tokenizer(context, return_tensors="pt").input_ids.to(model.device)
        stopping_criteria = StoppingCriteriaList([StopOnSubstring(tokenizer, input_ids.shape[1], STOP_MARKER)])
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )
        return tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

    return generate_fn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="training/outputs/stage4_agent_merged")
    parser.add_argument("--company-id", required=True)
    parser.add_argument("--max-turns", type=int, default=4)
    args = parser.parse_args()

    company = load_company(args.company_id)  # fail fast on a bad company_id
    user_prompt = f"请分析{company.name}({company.ticker})的风险状况,写一份风险报告。"

    generate_fn = _build_hf_generate_fn(args.model_path)
    print(run_agent(generate_fn, user_prompt, max_turns=args.max_turns))


if __name__ == "__main__":
    main()
