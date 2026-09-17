"""Transcript renderers: plain text, SubRip subtitles and JSON."""

from __future__ import annotations

import json
from collections.abc import Sequence

from echoes.schema import Utterance, to_dicts

FORMATS = ("txt", "srt", "json")


def format_timestamp(seconds: float, millis: bool = False) -> str:
    """HH:MM:SS, or HH:MM:SS,mmm (SubRip style) when `millis` is set."""
    total_ms = round(seconds * 1000)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    stamp = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{stamp},{ms:03d}" if millis else stamp


def to_txt(utterances: Sequence[Utterance]) -> str:
    return "".join(
        f"[{format_timestamp(u.start)} --> {format_timestamp(u.end)}] {u.speaker}: {u.text}\n\n"
        for u in utterances
    )


def to_srt(utterances: Sequence[Utterance]) -> str:
    return "\n".join(
        f"{i}\n"
        f"{format_timestamp(u.start, millis=True)} --> {format_timestamp(u.end, millis=True)}\n"
        f"[{u.speaker}] {u.text}\n"
        for i, u in enumerate(utterances, start=1)
    )


def to_json(utterances: Sequence[Utterance]) -> str:
    speakers = list(dict.fromkeys(u.speaker for u in utterances))
    return json.dumps(
        {"speakers": speakers, "utterances": to_dicts(utterances)}, ensure_ascii=False, indent=2
    )


def render(fmt: str, sentences: Sequence[Utterance], turns: Sequence[Utterance]) -> str:
    """Render a transcript format.

    Subtitles use one cue per sentence; the other formats use speaker turns
    (consecutive sentences of the same speaker merged together).
    """
    if fmt == "txt":
        return to_txt(turns)
    if fmt == "srt":
        return to_srt(sentences)
    if fmt == "json":
        return to_json(turns)
    raise ValueError(f"Unknown format: {fmt}")
