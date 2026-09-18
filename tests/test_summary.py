import json
import urllib.error

import pytest
from conftest import http_error

from echoes import summary
from echoes.summary import (
    MAX_ATTEMPTS,
    PROVIDERS,
    Summary,
    SummaryError,
    context_size,
    make_client,
    summarize,
    system_prompt,
)

TRANSCRIPT = "[00:00:00 --> 00:00:04] Speaker 1: What is your job?\n\n"

ANSWERS = {
    "gemini": {
        "candidates": [
            {
                "content": {"parts": [{"text": "hmm", "thought": True}, {"text": "# Summary"}]},
                "finishReason": "STOP",
            }
        ],
        "modelVersion": "gemini-test",
    },
    "openai": {
        "model": "gpt-test",
        "choices": [{"message": {"content": "# Summary"}, "finish_reason": "stop"}],
    },
    "anthropic": {
        "model": "claude-test",
        "content": [{"type": "text", "text": "# Summary"}],
        "stop_reason": "end_turn",
    },
    "mistral": {
        "model": "mistral-test",
        "choices": [{"message": {"content": "# Summary"}, "finish_reason": "stop"}],
    },
    "ollama": {
        "model": "qwen-test",
        "message": {"role": "assistant", "content": "<think>hmm</think>\n# Summary"},
        "done_reason": "stop",
    },
}

URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent",
    "openai": "https://api.openai.com/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
    "ollama": "http://localhost:11434/api/chat",
}


@pytest.mark.parametrize("name", list(PROVIDERS))
def test_providers(name, fake_llm, monkeypatch):
    provider = PROVIDERS[name]
    if provider.key_vars:
        monkeypatch.setenv(provider.key_vars[0], "secret-key")
    fake_llm.reply(ANSWERS[name])
    client = make_client(name, model=None if provider.default_model else "qwen3")

    result = summarize(TRANSCRIPT, client, language="fr")

    assert result.text == "# Summary"
    assert result.provider == name
    assert result.model.endswith("-test")
    request = fake_llm.requests[0]
    assert request.full_url == URLS[name]
    body = json.dumps(fake_llm.body())
    assert "Speaker 1: What is your job?" in body
    assert '\\"fr\\"' in body
    headers = dict(request.header_items())
    assert ("secret-key" in str(headers)) == bool(provider.key_vars)


def test_missing_key_says_where_to_put_it():
    with pytest.raises(SummaryError) as error:
        make_client("gemini")
    assert "GEMINI_API_KEY=your-key" in str(error.value)
    assert ".env" in str(error.value)
    assert "https://aistudio.google.com/apikey" in str(error.value)


def test_alternative_and_explicit_keys(monkeypatch):
    assert make_client("openai", api_key="explicit").api_key == "explicit"
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    assert make_client("gemini").api_key == "google-key"


def test_key_is_hidden_from_repr():
    assert "secret" not in repr(make_client("openai", api_key="secret"))


def test_ollama_needs_a_model():
    with pytest.raises(SummaryError, match="ollama list"):
        make_client("ollama")


def test_unknown_provider():
    with pytest.raises(SummaryError, match="Unknown summary provider"):
        make_client("acme")


def test_custom_address_needs_no_key(fake_llm):
    fake_llm.reply(ANSWERS["openai"])
    client = make_client("openai", model="local-model", base_url="http://localhost:1234/v1/")

    summarize(TRANSCRIPT, client)

    request = fake_llm.requests[0]
    assert request.full_url == "http://localhost:1234/v1/chat/completions"
    assert "Authorization" not in dict(request.header_items())


def test_transient_errors_are_retried(fake_llm):
    fake_llm.reply(http_error(503, {"error": {"message": "overloaded"}}), ANSWERS["gemini"])
    result = summarize(TRANSCRIPT, make_client("gemini", api_key="k"))
    assert result.text == "# Summary"
    assert len(fake_llm.requests) == 2


def test_retries_give_up(fake_llm):
    exceeded = {"error": {"message": "quota exceeded"}}
    fake_llm.reply(*(http_error(429, exceeded) for _ in range(MAX_ATTEMPTS)))
    with pytest.raises(SummaryError, match="quota exceeded") as error:
        summarize(TRANSCRIPT, make_client("gemini", api_key="k"))
    assert error.value.status == 429
    assert "echoes summarize" in str(error.value)
    assert len(fake_llm.requests) == MAX_ATTEMPTS


PER_DAY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
PER_MINUTE = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"


def quota_error(quota_id, retry_delay="22.5s"):
    return {
        "error": {
            "code": 429,
            "message": "You exceeded your current quota, please check your plan.",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [
                        {
                            "quotaId": quota_id,
                            "quotaDimensions": {"model": "gemini-9-flash"},
                            "quotaValue": "20",
                        }
                    ],
                },
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay},
            ],
        }
    }


def test_daily_quota_is_not_retried(fake_llm):
    fake_llm.reply(http_error(429, quota_error(PER_DAY)))
    with pytest.raises(SummaryError) as error:
        summarize(TRANSCRIPT, make_client("gemini", api_key="k"))
    message = str(error.value)
    assert f"{PER_DAY} (limit 20, model gemini-9-flash)" in message
    assert "retry tomorrow" in message
    assert len(fake_llm.requests) == 1


def test_suggested_retry_delay_is_used(fake_llm, monkeypatch):
    delays = []
    monkeypatch.setattr(summary.time, "sleep", delays.append)
    fake_llm.reply(http_error(429, quota_error(PER_MINUTE)), ANSWERS["gemini"])
    summarize(TRANSCRIPT, make_client("gemini", api_key="k"))
    assert delays == [23]


def test_client_errors_are_not_retried(fake_llm):
    fake_llm.reply(http_error(400, {"error": {"code": 400, "message": "API key not valid."}}))
    with pytest.raises(SummaryError, match="HTTP 400: API key not valid"):
        summarize(TRANSCRIPT, make_client("gemini", api_key="k"))
    assert len(fake_llm.requests) == 1


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"error": 'model "x" not found'}, 'model "x" not found'),
        ({"type": "error", "error": {"type": "auth", "message": "invalid x-api-key"}}, "invalid x-api-key"),
        ({"message": "Unauthorized"}, "Unauthorized"),
        ([{"error": {"message": "listed"}}], "listed"),
        ("<html>Bad gateway</html>", "<html>Bad gateway</html>"),
    ],
)
def test_error_messages(fake_llm, body, message):
    fake_llm.reply(http_error(404, body))
    with pytest.raises(SummaryError) as error:
        summarize(TRANSCRIPT, make_client("ollama", model="x"))
    assert message in str(error.value)


def test_unreachable_server(fake_llm):
    fake_llm.reply(urllib.error.URLError("Connection refused"))
    with pytest.raises(SummaryError, match="Could not reach Ollama at http://localhost:11434"):
        summarize(TRANSCRIPT, make_client("ollama", model="x"))


def test_blocked_answer(fake_llm):
    fake_llm.reply({"candidates": [{"finishReason": "SAFETY"}]})
    with pytest.raises(SummaryError, match="SAFETY"):
        summarize(TRANSCRIPT, make_client("gemini", api_key="k"))


def test_empty_answer(fake_llm):
    fake_llm.reply({"model": "m", "choices": [{"message": {"content": "  "}}]})
    with pytest.raises(SummaryError, match="empty summary"):
        summarize(TRANSCRIPT, make_client("openai", api_key="k"))


def test_empty_transcript_is_not_sent(fake_llm):
    with pytest.raises(SummaryError, match="empty"):
        summarize(" \n", make_client("openai", api_key="k"))
    assert not fake_llm.requests


def test_prompt_language():
    french = system_prompt("fr")
    assert '"fr"' in french
    assert "## Points clés" in french
    assert "Translate the headings" not in french

    unknown = system_prompt("nl")
    assert "## Key points" in unknown
    assert "Translate the headings" in unknown

    assert "language of the transcript" in system_prompt(None)


def test_prompt_gives_the_end_of_the_conversation():
    transcript = "[00:00:00 --> 00:00:04] A: Hi.\n\n[01:42:50 --> 01:43:05] B: Bye.\n\n"
    assert "from its beginning to its end, at 01:43:05:" in system_prompt("en", transcript)
    assert "from its beginning to its end:" in system_prompt("en", "No timestamps")


def test_ollama_context_fits_the_transcript(fake_llm):
    assert context_size("short") == 12288
    long_text = "x" * 300_000
    assert context_size(long_text) >= 100_000 + 8192
    assert context_size(long_text) % 4096 == 0

    fake_llm.reply(ANSWERS["ollama"])
    summarize(long_text, make_client("ollama", model="x"))
    assert fake_llm.body()["options"]["num_ctx"] >= 100_000


def test_markdown_front_matter():
    text = Summary("# Summary", "gemini", "gemini-test", created="2026-09-17T10:00:00").to_markdown()
    assert text == (
        "---\nprovider: gemini\nmodel: gemini-test\ncreated: 2026-09-17T10:00:00\n---\n\n# Summary\n"
    )
