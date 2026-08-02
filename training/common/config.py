"""Tiny YAML config loader shared by the stage scripts."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def patch_tokenizer_extra_special_tokens_list_bug() -> None:
    """Some HF-hosted tokenizer_config.json files already store
    `extra_special_tokens` in transformers v5's dict format serialized as a
    plain list; transformers<5's `_set_model_specific_special_tokens` calls
    `.keys()` on it unconditionally and crashes ("'list' object has no
    attribute 'keys'"). Normalize list -> dict before it gets there.
    See https://github.com/huggingface/transformers/issues/45376.
    """
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if getattr(PreTrainedTokenizerBase._set_model_specific_special_tokens, "_patched_list_bug", False):
        return  # idempotent - safe to call from multiple entry points

    original = PreTrainedTokenizerBase._set_model_specific_special_tokens

    def patched(self, special_tokens):
        if isinstance(special_tokens, list):
            special_tokens = {tok: tok for tok in special_tokens}
        return original(self, special_tokens)

    patched._patched_list_bug = True
    PreTrainedTokenizerBase._set_model_specific_special_tokens = patched
