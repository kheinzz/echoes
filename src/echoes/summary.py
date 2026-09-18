"""Optional summary of the conversation with a large language model (LLM).

The transcript text (never the audio) is sent to the chosen provider, except
with Ollama, which runs on your own machine. Only the standard library is used:
no provider SDK is needed.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

log = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 8192
MAX_ATTEMPTS = 3
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
HTTP_HINTS = {
    400: "check your API key and the model name",
    401: "check your API key",
    403: "check your API key and its permissions",
    404: "check the model name and the API address",
    429: "rate limit reached: retry later with `echoes summarize`",
    503: "the model is busy: retry later with `echoes summarize`, or choose another model",
}
DAILY_QUOTA_HINT = (
    "daily quota reached for this model: retry tomorrow, or choose another model or plan"
)
# Reasoning models served by some OpenAI-compatible APIs prepend their thoughts
THINKING = re.compile(r"^\s*<think>.*?</think>", re.DOTALL)

SYSTEM_PROMPT = """\
You summarize transcripts of recorded conversations: research interviews, \
meetings, podcasts. The transcript was produced automatically. Each paragraph \
starts with a time range and a speaker label. Speaker labels come from an \
automatic speaker detection and can occasionally be wrong, and some words can \
be misrecognized.

First, decide which kind of conversation it is:
- an interview: one person mainly asks questions and the others answer them;
- an open discussion: there is no clear question-and-answer structure.

Then write the summary in Markdown, {language}, using the matching layout \
below.{translate}

Layout for an interview:

# {summary}
One short paragraph: who is interviewed (role, organization, if stated), the \
subject and the main takeaways.

## {questions}
### 1. <the question, rephrased concisely> (<start time, HH:MM:SS>)
The answer, summarized: key facts, figures, names, examples and opinions. \
Short quotes when they are telling.

(One section per question, in order. Merge follow-up questions and requests \
for clarification into the question they relate to.)

## {key_points}
- The most important facts and ideas of the interview.

Layout for an open discussion:

# {summary}
One short paragraph: the participants (if identifiable), the subject and the \
main takeaways.

## {topics}
### <topic>
What was said about it, and by whom.

## {decisions}
- Only if the discussion contains any; otherwise leave this section out.

Rules:
- Cover the whole conversation, from its beginning to its end{end}: a long \
interview usually has many questions, and its last part matters as much as its \
first one.
- Use only information from the transcript. Do not add outside knowledge or \
invent anything; when a passage is unclear, say so.
- Refer to participants by their labels (e.g. "Speaker 1"), unless their name \
or role is clearly stated in the conversation.
- Be concise, but keep concrete facts, figures, names and dates.
- Answer with the Markdown summary only, without any introduction.
"""
# Headings given ready-made: small models tend to copy English ones verbatim
HEADINGS = {
    "en": ("Summary", "Questions", "Key points", "Topics", "Decisions and next steps"),
    "fr": ("Résumé", "Questions", "Points clés", "Thèmes", "Décisions et prochaines étapes"),
    "es": ("Resumen", "Preguntas", "Puntos clave", "Temas", "Decisiones y próximos pasos"),
    "de": ("Zusammenfassung", "Fragen", "Kernpunkte", "Themen", "Entscheidungen und nächste Schritte"),
    "it": ("Sintesi", "Domande", "Punti chiave", "Temi", "Decisioni e prossimi passi"),
    "pt": ("Resumo", "Perguntas", "Pontos-chave", "Temas", "Decisões e próximos passos"),
}
TIMESTAMP = re.compile(r"--> (\d{2}:\d{2}:\d{2})")


class SummaryError(RuntimeError):
    """The summary could not be produced."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status  # HTTP status of the provider's answer, if any


@dataclass(frozen=True)
class Provider:
    name: str
    label: str
    default_model: str | None
    base_url: str
    key_vars: tuple[str, ...]
    key_url: str | None
    call: Callable[[Client, str, str], tuple[str, str | None]]
    local: bool = False
    timeout: float = 600


@dataclass(frozen=True)
class Client:
    """Checked settings for one provider, see `make_client`."""

    provider: Provider
    model: str
    url: str
    api_key: str | None = field(default=None, repr=False)


@dataclass
class Summary:
    text: str
    provider: str
    model: str
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_markdown(self) -> str:
        """The summary, with its provenance as YAML front matter."""
        return (
            f"---\nprovider: {self.provider}\nmodel: {self.model}\ncreated: {self.created}\n---\n\n"
            f"{self.text}\n"
        )


def make_client(
    provider: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Client:
    """Check the summary settings, so that errors show up before the long steps."""
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise SummaryError(
            f"Unknown summary provider {provider!r} (choose among {', '.join(PROVIDERS)})"
        )
    model = model or spec.default_model
    if not model:
        raise SummaryError(
            f"{spec.label} has no default model: choose one of your local models "
            "(see `ollama list`) with --summary-model, or --model with `echoes summarize`"
        )
    api_key = api_key or find_api_key(spec)
    # A custom address may be a local server that doesn't need any key
    if not api_key and spec.key_vars and not base_url:
        raise SummaryError(missing_key_help(spec))
    return Client(spec, model, (base_url or spec.base_url).rstrip("/"), api_key)


def summarize(transcript: str, client: Client, *, language: str | None = None) -> Summary:
    """Summarize a transcript rendered as text (see `export.to_txt`)."""
    if not transcript.strip():
        raise SummaryError("The transcript is empty: nothing to summarize")
    spec = client.provider
    host = urllib.parse.urlsplit(client.url).netloc
    if spec.local:
        log.info("Summarizing with %s model '%s' on %s", spec.label, client.model, host)
    else:
        log.info("Sending the transcript to %s for the summary (model '%s')", host, client.model)
    system = system_prompt(language, transcript)
    text, version = spec.call(client, system, f"Transcript:\n\n{transcript}")
    text = THINKING.sub("", text).strip()
    if not text:
        raise SummaryError(f"{spec.label} returned an empty summary")
    return Summary(text=text, provider=spec.name, model=version or client.model)


def system_prompt(language: str | None, transcript: str = "") -> str:
    if language:
        target = f'in the language with ISO 639-1 code "{language}"'
    else:
        target = "in the language of the transcript"
    headings = HEADINGS.get(language or "")
    translate = "" if headings else " Translate the headings into that language."
    summary, questions, key_points, topics, decisions = headings or HEADINGS["en"]
    # An explicit end time keeps models from stopping halfway through long recordings
    times = TIMESTAMP.findall(transcript)
    return SYSTEM_PROMPT.format(
        language=target,
        translate=translate,
        summary=summary,
        questions=questions,
        key_points=key_points,
        topics=topics,
        decisions=decisions,
        end=f", at {times[-1]}" if times else "",
    )


def find_api_key(spec: Provider) -> str | None:
    for var in spec.key_vars:
        value = os.environ.get(var, "").strip()
        if value:
            return value
    return None


def missing_key_help(spec: Provider) -> str:
    var = spec.key_vars[0]
    return (
        f"No API key found for {spec.label}.\n"
        f"  1. create a key at {spec.key_url}\n"
        "  2. add this line to the .env file of the folder where you run echoes\n"
        "     (or to the file given with --env-file):\n"
        f"       {var}=your-key\n"
        f"     or set the {var} environment variable"
    )


def call_gemini(client: Client, system: str, prompt: str) -> tuple[str, str | None]:
    data = post_json(
        f"{client.url}/models/{client.model}:generateContent",
        {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        },
        {"x-goog-api-key": client.api_key},
        client,
    )
    candidate = (data.get("candidates") or [{}])[0]
    reason = candidate.get("finishReason") or data.get("promptFeedback", {}).get("blockReason")
    parts = candidate.get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
    if reason == "MAX_TOKENS":
        warn_truncated(client)
    elif not text and reason:
        raise SummaryError(f"{client.provider.label} returned no summary (reason: {reason})")
    return text, data.get("modelVersion")


def call_openai(client: Client, system: str, prompt: str) -> tuple[str, str | None]:
    """OpenAI chat completions, also spoken by Mistral, OpenRouter, Groq, LM Studio..."""
    data = post_json(
        f"{client.url}/chat/completions",
        {
            "model": client.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        },
        {"Authorization": f"Bearer {client.api_key}" if client.api_key else None},
        client,
    )
    choice = (data.get("choices") or [{}])[0]
    if choice.get("finish_reason") == "length":
        warn_truncated(client)
    return choice.get("message", {}).get("content") or "", data.get("model")


def call_anthropic(client: Client, system: str, prompt: str) -> tuple[str, str | None]:
    data = post_json(
        f"{client.url}/messages",
        {
            "model": client.model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        },
        {"x-api-key": client.api_key, "anthropic-version": "2023-06-01"},
        client,
    )
    if data.get("stop_reason") == "max_tokens":
        warn_truncated(client)
    text = "".join(
        block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
    )
    return text, data.get("model")


def call_ollama(client: Client, system: str, prompt: str) -> tuple[str, str | None]:
    data = post_json(
        f"{client.url}/api/chat",
        {
            "model": client.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {"num_ctx": context_size(system, prompt)},
        },
        {},
        client,
    )
    if data.get("done_reason") == "length":
        warn_truncated(client)
    return data.get("message", {}).get("content") or "", data.get("model")


def context_size(*texts: str) -> int:
    """Context window to request from Ollama.

    Its default window is often too small for a whole interview, and Ollama
    silently drops the beginning of prompts that don't fit.
    """
    tokens = sum(len(text) for text in texts) // 3 + MAX_OUTPUT_TOKENS
    return max(8192, math.ceil(tokens / 4096) * 4096)


def warn_truncated(client: Client) -> None:
    log.warning("The %s answer was cut off: the summary is incomplete", client.provider.label)


def post_json(url: str, payload: dict, headers: dict, client: Client) -> dict:
    """POST a JSON payload and return the JSON answer, retrying transient errors."""
    label = client.provider.label
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", **{k: v for k, v in headers.items() if v}}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=client.provider.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            error = read_error(exc)
            # a daily quota won't come back before tomorrow: don't wait for it
            retry = exc.code in RETRY_STATUSES and not error.daily_quota
            if retry and attempt < MAX_ATTEMPTS:
                delay = retry_delay(exc, error, attempt)
                log.warning(
                    "%s answered HTTP %d (%s): retrying in %ds",
                    label, exc.code, error.message, delay,
                )
                time.sleep(delay)
                continue
            hint = DAILY_QUOTA_HINT if error.daily_quota else HTTP_HINTS.get(exc.code)
            raise SummaryError(
                f"{label} answered HTTP {exc.code}: {error.message}"
                + (f"\n({hint})" if hint else ""),
                status=exc.code,
            ) from exc
        except OSError as exc:
            reason = getattr(exc, "reason", None) or exc
            raise SummaryError(f"Could not reach {label} at {client.url}: {reason}") from exc
        except ValueError as exc:
            raise SummaryError(f"{label} sent an answer that is not valid JSON") from exc
    raise AssertionError("unreachable")


@dataclass
class ApiError:
    message: str
    daily_quota: bool = False
    retry_after: float | None = None


def read_error(exc: urllib.error.HTTPError) -> ApiError:
    """Extract what matters from the error body sent by the provider."""
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except OSError:
        body = ""
    try:
        data = json.loads(body)
    except ValueError:
        return ApiError(shorten(body) or str(exc.reason))
    if isinstance(data, list) and data:
        data = data[0]
    error = data.get("error", data) if isinstance(data, dict) else data
    if not isinstance(error, dict):
        return ApiError(shorten(str(error)))
    result = ApiError(shorten(str(error.get("message") or error.get("detail") or error)))
    # Google APIs detail which quota is exceeded, and when to retry
    quotas = []
    for detail in error.get("details") or []:
        kind = str(detail.get("@type", "")) if isinstance(detail, dict) else ""
        if kind.endswith("QuotaFailure"):
            for violation in detail.get("violations") or []:
                quota = str(violation.get("quotaId", "unknown"))
                model = (violation.get("quotaDimensions") or {}).get("model")
                limit = violation.get("quotaValue")
                quotas.append(f"{quota} (limit {limit}" + (f", model {model})" if model else ")"))
                result.daily_quota = result.daily_quota or "PerDay" in quota
        elif kind.endswith("RetryInfo"):
            result.retry_after = parse_seconds(detail.get("retryDelay"))
    if quotas:
        result.message = "quota exceeded: " + "; ".join(quotas)
    return result


def shorten(text: str) -> str:
    return " ".join(text.split())[:500]


def parse_seconds(value: object) -> float | None:
    """Parse a protobuf duration such as "22.5s"."""
    try:
        return float(str(value).removesuffix("s"))
    except ValueError:
        return None


def retry_delay(exc: urllib.error.HTTPError, error: ApiError, attempt: int) -> int:
    try:
        seconds = float((exc.headers or {}).get("Retry-After", ""))
    except ValueError:
        seconds = error.retry_after or 10 * 2 ** (attempt - 1)
    return min(math.ceil(seconds), 60)


PROVIDERS = {
    provider.name: provider
    for provider in (
        Provider(
            name="gemini",
            label="Google Gemini",
            # an alias: it follows the current Flash model, which pinned names don't
            default_model="gemini-flash-latest",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            key_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            key_url="https://aistudio.google.com/apikey",
            call=call_gemini,
        ),
        Provider(
            name="openai",
            label="OpenAI",
            default_model="gpt-5-mini",
            base_url="https://api.openai.com/v1",
            key_vars=("OPENAI_API_KEY",),
            key_url="https://platform.openai.com/api-keys",
            call=call_openai,
        ),
        Provider(
            name="anthropic",
            label="Anthropic",
            default_model="claude-sonnet-5",
            base_url="https://api.anthropic.com/v1",
            key_vars=("ANTHROPIC_API_KEY",),
            key_url="https://console.anthropic.com/settings/keys",
            call=call_anthropic,
        ),
        Provider(
            name="mistral",
            label="Mistral AI",
            default_model="mistral-medium-latest",
            base_url="https://api.mistral.ai/v1",
            key_vars=("MISTRAL_API_KEY",),
            key_url="https://console.mistral.ai/api-keys",
            call=call_openai,
        ),
        Provider(
            name="ollama",
            label="Ollama",
            default_model=None,
            base_url="http://localhost:11434",
            key_vars=(),
            key_url=None,
            call=call_ollama,
            local=True,
            # local models can be slow, especially on CPU
            timeout=3600,
        ),
    )
}
