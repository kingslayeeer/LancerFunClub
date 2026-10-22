import pytest

from llm_module import LLMError, edit_text, fill_template, generate_document


class StubLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kw):
        self.calls.append((prompt, kw))
        return "ok"


ENTRY = {
    "id": "letter_test",
    "meta": {"length": "long"},
    "prompt": {
        "system": "sys",
        "user_template": "Письмо.\n- Тема: {{topic}}\n- Договор: {{contract_number}}\n- Суть: {{body}}\nКонец.",
        "few_shot": [{"user": "u1", "assistant": "a1", "note": "x"}],
    },
}

EDITOR = {
    "prompt": {
        "system": "ed",
        "user_template": "Стиль: {{style}}\nОбъём: {{length}}\nТекст:\n{{text}}",
        "few_shot": [],
    }
}


def test_fill_empty_field_placeholder():
    s = fill_template(ENTRY["prompt"]["user_template"], {"topic": "оплата", "body": "долг", "contract_number": " "})
    assert "- Договор: [НОМЕР ДОГОВОРА]" in s
    assert "- Тема: оплата" in s and "- Суть: долг" in s
    assert "{{" not in s


def test_fill_unknown_field():
    assert fill_template("x {{foo}}", {}) == "x [FOO]"


def test_generate_document():
    llm = StubLLM()
    generate_document(ENTRY, {"topic": "t", "body": "b", "contract_number": "12"}, llm=llm)
    prompt, kw = llm.calls[0]
    assert "- Договор: 12" in prompt
    assert kw["system"] == "sys"
    assert kw["examples"] == [("u1", "a1")]
    assert kw["max_tokens"] == 1600


def test_generate_document_needs_body():
    with pytest.raises(LLMError):
        generate_document(ENTRY, {"topic": "t"}, llm=StubLLM())


def test_edit_text():
    llm = StubLLM()
    edit_text("Доводим до вашего сведения...", style="official", editor=EDITOR, llm=llm)
    prompt, kw = llm.calls[0]
    assert "официально-деловой" in prompt and "стандартный" in prompt
    assert "Доводим до вашего сведения" in prompt
    assert kw["preset"] == "edit"
