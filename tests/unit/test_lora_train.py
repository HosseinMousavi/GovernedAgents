"""train.py: only the assistant turn is learned (prompt tokens are masked with -100)."""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")

from store_support.lora.train import _encode  # noqa: E402


class FakeTok:
    eos_token = "<e>"  # noqa: S105 - not a secret

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):  # noqa: ANN001, ANN201
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):  # noqa: ANN001, ANN201
        return {"input_ids": [ord(c) for c in text]}


def test_prompt_tokens_are_masked_and_answer_tokens_are_labelled() -> None:
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "[0]"},
    ]
    enc = _encode(FakeTok(), msgs, max_len=4096)
    prompt_len = len("<system>s<user>u<assistant>")
    assert enc["labels"][:prompt_len] == [-100] * prompt_len
    assert enc["labels"][prompt_len:] == [ord(c) for c in "[0]<e>"]
    assert len(enc["input_ids"]) == len(enc["labels"]) == len(enc["attention_mask"])


def test_truncation_keeps_label_alignment() -> None:
    msgs = [{"role": "user", "content": "x" * 50}, {"role": "assistant", "content": "y" * 50}]
    enc = _encode(FakeTok(), msgs, max_len=60)
    assert len(enc["input_ids"]) == 60 and len(enc["labels"]) == 60
