import io
import json
import os
import urllib.error
import urllib.request

import pytest

from echoes import summary

SECRET_VARS = {"HF_TOKEN"} | {var for p in summary.PROVIDERS.values() for var in p.key_vars}


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Start without API keys, and keep variables loaded from .env files from leaking."""
    environ = {name: value for name, value in os.environ.items() if name not in SECRET_VARS}
    monkeypatch.setattr(os, "environ", environ)


class FakeLLM:
    """Stands in for urlopen: records the requests and plays the given answers."""

    def __init__(self, monkeypatch):
        self.answers = []
        self.requests = []
        monkeypatch.setattr(urllib.request, "urlopen", self.urlopen)
        monkeypatch.setattr(summary.time, "sleep", lambda seconds: None)

    def reply(self, *answers):
        self.answers.extend(answers)
        return self

    def urlopen(self, request, timeout):
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return io.BytesIO(json.dumps(answer).encode())

    def body(self, index=-1):
        return json.loads(self.requests[index].data)


@pytest.fixture
def fake_llm(monkeypatch):
    return FakeLLM(monkeypatch)


def http_error(code, body):
    raw = body if isinstance(body, str) else json.dumps(body)
    return urllib.error.HTTPError("https://api", code, "Error", {}, io.BytesIO(raw.encode()))


def gemini_answer(text):
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
        "modelVersion": "gemini-test",
    }
