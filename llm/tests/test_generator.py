import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llm_module import LLMError, LLMGenerator, LLMTimeoutError, LLMUnavailableError


class FakeHandler(BaseHTTPRequestHandler):
    last_body: dict = {}
    mode = "ok"

    def log_message(self, *a):
        pass

    def _send(self, code, data):
        raw = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send(200, {"models": [{"name": "qwen3:8b"}]})
        elif self.path == "/v1/models":
            self._send(200, {"data": []})
        else:
            self._send(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeHandler.last_body = body
        if FakeHandler.mode == "slow":
            time.sleep(2)
        if FakeHandler.mode == "error":
            return self._send(500, {"error": "boom"})
        if body.get("model") == "missing":
            return self._send(404, {"error": "model not found"})
        content = "  Уважаемый коллега!  "
        if FakeHandler.mode == "think":
            content = "<think>размышляю</think>\nТекст письма"
        if self.path == "/api/chat":
            self._send(200, {"model": body["model"], "message": {"content": content},
                             "prompt_eval_count": 20, "eval_count": 10})
        elif self.path == "/v1/chat/completions":
            self._send(200, {"model": body["model"],
                             "choices": [{"message": {"content": content}}],
                             "usage": {"prompt_tokens": 20, "completion_tokens": 10}})


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), FakeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


@pytest.fixture(autouse=True)
def reset_mode():
    FakeHandler.mode = "ok"


def test_ollama_ok(server):
    llm = LLMGenerator(model="qwen3:8b", base_url=server, retries=0)
    r = llm.generate("Привет", system="Роль")
    assert r.text == "Уважаемый коллега!"
    assert r.output_tokens == 10
    body = FakeHandler.last_body
    assert body["think"] is False
    assert body["messages"][0]["role"] == "system"
    assert body["options"]["temperature"] == 0.4


def test_examples(server):
    llm = LLMGenerator(model="qwen3:8b", base_url=server, retries=0)
    llm.generate("вопрос", system="роль", examples=[("пример", "ответ")])
    roles = [m["role"] for m in FakeHandler.last_body["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_overrides_and_preset(server):
    llm = LLMGenerator(model="qwen3:8b", base_url=server, retries=0)
    llm.generate("x", preset="classify", max_tokens=50)
    opts = FakeHandler.last_body["options"]
    assert opts["temperature"] == 0.0 and opts["num_predict"] == 50


def test_openai_backend(server):
    llm = LLMGenerator(model="any", backend="openai", base_url=server, retries=0)
    r = llm.generate("Привет")
    assert r.text == "Уважаемый коллега!"
    assert FakeHandler.last_body["max_tokens"] == 1024


def test_think_removed(server):
    FakeHandler.mode = "think"
    r = LLMGenerator(model="qwen3:8b", base_url=server, retries=0).generate("x")
    assert r.text == "Текст письма"


def test_missing_model(server):
    with pytest.raises(LLMUnavailableError):
        LLMGenerator(model="missing", base_url=server).generate("x")


def test_server_down():
    with pytest.raises(LLMUnavailableError):
        LLMGenerator(base_url="http://127.0.0.1:1", retries=0).generate("x")


def test_timeout(server):
    FakeHandler.mode = "slow"
    with pytest.raises(LLMTimeoutError):
        LLMGenerator(model="qwen3:8b", base_url=server, timeout=0.5, retries=0).generate("x")


def test_server_error_retries(server):
    FakeHandler.mode = "error"
    with pytest.raises(LLMError):
        LLMGenerator(model="qwen3:8b", base_url=server, retries=0).generate("x")


def test_empty_prompt():
    with pytest.raises(LLMError):
        LLMGenerator().generate("   ")


def test_is_available(server):
    assert LLMGenerator(model="qwen3:8b", base_url=server).is_available()
    assert LLMGenerator(model="qwen3", base_url=server).is_available()
    assert not LLMGenerator(model="gemma3", base_url=server).is_available()
