import json
import re
from pathlib import Path

from .generator import LLMError, LLMGenerator

# llm/llm_module/prompt_runner.py -> корень репо -> prompts/
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
PROMPTS_FILE = PROMPTS_DIR / "data" / "prompts.json"
EDITOR_FILE = PROMPTS_DIR / "editor.json"

STYLE_NAMES = {
    "official": "официально-деловой",
    "neutral": "нейтрально-деловой",
    "friendly": "дружелюбно-деловой",
}
LENGTH_NAMES = {"short": "краткий", "standard": "стандартный", "long": "развёрнутый"}

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def load_prompts(path=PROMPTS_FILE):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {p["id"]: p for p in data["prompts"]}


def load_editor(path=EDITOR_FILE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# если поле не заполнено, ставим плейсхолдер в [] как у Участника 2 ([ФИО], [ДАТА]),
# чтобы модель не выдумывала номер договора, сумму и т.п.
FIELD_LABELS = {
    "topic": "ТЕМА",
    "recipient": "АДРЕСАТ",
    "organization": "ОРГАНИЗАЦИЯ",
    "contract_number": "НОМЕР ДОГОВОРА",
    "amount": "СУММА",
    "deadline": "СРОК",
    "reason": "ОСНОВАНИЕ",
    "sender": "ОТПРАВИТЕЛЬ",
}


def fill_template(template, fields):
    def repl(m):
        key = m.group(1)
        value = str(fields.get(key) or "").strip()
        if value:
            return value
        return "[" + FIELD_LABELS.get(key, key.upper()) + "]"

    return _PLACEHOLDER.sub(repl, template)


def get_examples(entry):
    return [(ex["user"], ex["assistant"]) for ex in entry["prompt"].get("few_shot", [])]


def _preset_for(length):
    if length == "short":
        return "short", {}
    if length == "long":
        return "document", {"max_tokens": 1600}
    return "document", {}


def generate_document(entry, fields, llm=None):
    """entry - один промпт из prompts.json, fields - {"topic": ..., "recipient": ..., ...}"""
    if not str(fields.get("body") or "").strip():
        raise LLMError("не заполнено поле body (суть документа)")
    llm = llm or LLMGenerator()
    preset, extra = _preset_for(entry["meta"].get("length"))
    return llm.generate(
        fill_template(entry["prompt"]["user_template"], fields),
        system=entry["prompt"]["system"],
        examples=get_examples(entry),
        preset=preset,
        **extra,
    )


def edit_text(text, style="neutral", length="standard", editor=None, llm=None):
    if not text or not text.strip():
        raise LLMError("пустой текст для редактирования")
    editor = editor or load_editor()
    llm = llm or LLMGenerator()
    fields = {
        "text": text,
        "style": STYLE_NAMES.get(style, style),
        "length": LENGTH_NAMES.get(length, length),
    }
    return llm.generate(
        fill_template(editor["prompt"]["user_template"], fields),
        system=editor["prompt"]["system"],
        examples=get_examples(editor),
        preset="edit",
    )
