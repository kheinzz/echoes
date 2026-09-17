"""Plain data structures shared by the pipeline steps."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    """A chunk of transcript as returned by Whisper."""

    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> Segment:
        words = [Word(**word) for word in data.get("words") or []]
        return cls(start=data["start"], end=data["end"], text=data["text"], words=words)


@dataclass
class Turn:
    """A time span during which a single speaker is talking."""

    start: float
    end: float
    speaker: str


@dataclass
class Utterance:
    """A piece of transcript attributed to one speaker."""

    start: float
    end: float
    speaker: str
    text: str


def to_dicts(items: Iterable) -> list[dict]:
    return [asdict(item) for item in items]
