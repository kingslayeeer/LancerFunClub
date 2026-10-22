import os

BACKEND = os.getenv("LLM_BACKEND", "ollama")  # ollama / openai
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("LLM_MODEL", "qwen3:8b")  # по бенчмарку быстрее gemma3:12b в 2 раза

TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT", "180"))  # на cpu бывает долго
RETRIES = int(os.getenv("LLM_RETRIES", "1"))
KEEP_ALIVE = os.getenv("LLM_KEEP_ALIVE", "30m")

_base = {"top_p": 0.9, "repeat_penalty": 1.1, "num_ctx": 8192, "stop": []}

PRESETS = {
    "document": {**_base, "temperature": 0.4, "max_tokens": 1024},
    "edit": {**_base, "temperature": 0.2, "max_tokens": 1024},
    "short": {**_base, "temperature": 0.3, "max_tokens": 400},
    # для Дани, json с параметрами
    "classify": {**_base, "temperature": 0.0, "max_tokens": 200, "num_ctx": 4096},
}
