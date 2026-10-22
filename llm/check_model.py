import sys

from llm_module import LLMError, LLMGenerator

model = sys.argv[1] if len(sys.argv) > 1 else None
llm = LLMGenerator(model=model)

print(llm.backend, llm.base_url, llm.model)
print("available:", llm.is_available())

try:
    r = llm.generate(
        "Напиши короткое письмо коллеге с просьбой прислать отчёт до пятницы.",
        system="Ты помощник по деловой переписке. Пиши кратко и вежливо.",
    )
except LLMError as e:
    print("ошибка:", e)
    sys.exit(1)

print("-" * 40)
print(r.text)
print("-" * 40)
print(f"{r.seconds:.1f} c, {r.output_tokens} токенов, {r.tokens_per_second:.1f} ток/с")
