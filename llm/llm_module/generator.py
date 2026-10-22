import time
from dataclasses import dataclass, field

import requests

from . import config


class LLMError(Exception):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMUnavailableError(LLMError):
    pass


@dataclass
class GenerationResult:
    text: str
    model: str
    seconds: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def tokens_per_second(self):
        if self.seconds <= 0 or self.output_tokens == 0:
            return 0.0
        return self.output_tokens / self.seconds


class LLMGenerator:
    def __init__(self, model=None, backend=None, base_url=None, timeout=None, retries=None):
        self.model = model or config.MODEL
        self.backend = backend or config.BACKEND
        self.base_url = (base_url or config.BASE_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else config.TIMEOUT_SECONDS
        self.retries = retries if retries is not None else config.RETRIES

        if self.backend not in ("ollama", "openai"):
            raise ValueError("backend: ollama или openai")

    def generate(self, prompt, system=None, preset="document", examples=None, **overrides):
        if not prompt or not prompt.strip():
            raise LLMError("пустой prompt")
        if preset not in config.PRESETS:
            raise LLMError(f"нет такого пресета: {preset}")

        params = dict(config.PRESETS[preset])
        params.update(overrides)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        # few-shot: список пар (вопрос, ответ)
        for user_msg, answer in examples or []:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": prompt})

        err = None
        for i in range(self.retries + 1):
            try:
                if self.backend == "ollama":
                    return self._ollama(messages, params)
                return self._openai(messages, params)
            except LLMUnavailableError:
                raise
            except LLMError as e:
                err = e
                if i < self.retries:
                    time.sleep(1.5 * (i + 1))
        raise err

    def is_available(self):
        try:
            if self.backend == "ollama":
                r = requests.get(self.base_url + "/api/tags", timeout=5)
                r.raise_for_status()
                for m in r.json().get("models", []):
                    if m["name"] == self.model or m["name"].split(":")[0] == self.model:
                        return True
                return False
            return requests.get(self.base_url + "/v1/models", timeout=5).ok
        except requests.RequestException:
            return False

    def _ollama(self, messages, params):
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,  # qwen3 иначе долго думает
            "keep_alive": config.KEEP_ALIVE,
            "options": {
                "temperature": params["temperature"],
                "top_p": params["top_p"],
                "repeat_penalty": params["repeat_penalty"],
                "num_predict": params["max_tokens"],
                "num_ctx": params["num_ctx"],
                "stop": params["stop"],
            },
        }
        data, sec = self._post("/api/chat", body)
        text = (data.get("message") or {}).get("content", "")
        return GenerationResult(
            text=clean(text),
            model=data.get("model", self.model),
            seconds=sec,
            prompt_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            raw=data,
        )

    # llama.cpp server / vllm
    def _openai(self, messages, params):
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "max_tokens": params["max_tokens"],
            "stop": params["stop"] or None,
        }
        data, sec = self._post("/v1/chat/completions", body)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError("странный ответ: " + str(data)[:200])
        usage = data.get("usage") or {}
        return GenerationResult(
            text=clean(text),
            model=data.get("model", self.model),
            seconds=sec,
            prompt_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            raw=data,
        )

    def _post(self, path, body):
        t = time.perf_counter()
        try:
            r = requests.post(self.base_url + path, json=body, timeout=(5, self.timeout))
        except requests.exceptions.ConnectionError:
            raise LLMUnavailableError(f"не подключиться к {self.base_url}, сервер запущен?")
        except requests.exceptions.Timeout:
            raise LLMTimeoutError(f"таймаут {self.timeout} с")
        sec = time.perf_counter() - t

        if r.status_code == 404:
            raise LLMUnavailableError(f"модели {self.model} нет (ollama pull {self.model})")
        if r.status_code >= 400:
            raise LLMError(f"ошибка {r.status_code}: {r.text[:200]}")
        try:
            return r.json(), sec
        except ValueError:
            raise LLMError("ответ не json")


def clean(text):
    text = text or ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()
