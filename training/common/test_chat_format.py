from training.common.chat_format import render_messages, render_prompt


def test_render_messages_includes_all_roles_in_order():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
        {"role": "assistant", "content": "asst"},
    ]
    text = render_messages(messages)
    assert text.index("<|system|>") < text.index("<|user|>") < text.index("<|assistant|>")
    assert "sys" in text and "usr" in text and "asst" in text


def test_render_messages_appends_eos_token():
    messages = [{"role": "user", "content": "hi"}]
    text = render_messages(messages, eos_token="<eos>")
    assert text.endswith("<eos>")


def test_render_messages_no_eos_by_default():
    messages = [{"role": "user", "content": "hi"}]
    text = render_messages(messages)
    assert not text.endswith("<eos>")
    assert text.endswith("hi")


def test_render_prompt_ends_with_assistant_tag_and_no_content():
    prompt = render_prompt("system text", "user text")
    assert prompt.rstrip().endswith("<|assistant|>")
    assert "system text" in prompt
    assert "user text" in prompt


def test_render_prompt_is_prefix_of_render_messages_for_same_content():
    system_prompt, user_prompt, assistant_text = "sys", "usr", "reply"
    prompt = render_prompt(system_prompt, user_prompt)
    full = render_messages(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_text},
        ]
    )
    # The assistant content should immediately follow the rendered prompt.
    assert full[len(prompt):].startswith(assistant_text)
