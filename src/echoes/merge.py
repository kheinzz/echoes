"""Align the Whisper transcript with the speaker turns.

Each sentence is attributed to the speaker who talks the most during its
words. Working at sentence level (rather than per Whisper segment) catches
speaker changes inside a segment, while avoiding sentences cut in half by
slightly misplaced turn boundaries.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable, Iterator

from echoes.schema import Segment, Turn, Utterance, Word

SENTENCE_END = (".", "?", "!", "…")
CLOSING_CHARS = "\"'»)]”’"
UNKNOWN_SPEAKER = "Unknown"
MIN_SPAN = 1e-3


class SpeakerIndex:
    """Fast lookup of the speakers active during a time span."""

    def __init__(self, turns: Iterable[Turn]):
        self.turns = sorted(turns, key=lambda turn: turn.start)
        self.starts = [turn.start for turn in self.turns]
        # Running maximum of the turn ends: every turn before the first index
        # whose running max exceeds t ends at or before t, even when turns
        # overlap each other
        self.max_ends: list[float] = []
        self.max_end_index: list[int] = []
        best = 0
        for i, turn in enumerate(self.turns):
            if turn.end > self.turns[best].end:
                best = i
            self.max_ends.append(self.turns[best].end)
            self.max_end_index.append(best)

    def speaker_weights(self, start: float, end: float) -> dict[str, float]:
        """Seconds of overlap per speaker between `start` and `end`.

        When no turn overlaps the span, the nearest turn gets it all.
        """
        if not self.turns:
            return {}
        end = max(end, start + MIN_SPAN)
        first = bisect_right(self.max_ends, start)
        stop = bisect_left(self.starts, end)
        weights: dict[str, float] = defaultdict(float)
        for turn in self.turns[first:stop]:
            overlap = min(end, turn.end) - max(start, turn.start)
            if overlap > 0:
                weights[turn.speaker] += overlap
        if not weights:
            weights[self._nearest(start, end, stop).speaker] = end - start
        return dict(weights)

    def _nearest(self, start: float, end: float, stop: int) -> Turn:
        # Nothing overlaps, so turns[:stop] all end before `start` and
        # turns[stop:] all begin after `end`
        candidates = []
        if stop > 0:
            before = self.turns[self.max_end_index[stop - 1]]
            candidates.append((start - before.end, before))
        if stop < len(self.turns):
            after = self.turns[stop]
            candidates.append((after.start - end, after))
        return min(candidates, key=lambda candidate: candidate[0])[1]


def split_sentences(segments: Iterable[Segment], max_pause: float = 1.0) -> Iterator[list[Word]]:
    """Group words into sentence-like units.

    A unit ends on sentence punctuation or before a pause longer than
    `max_pause` seconds. Standalone punctuation (such as a closing "»") stays
    with the preceding words. A segment without word timestamps (e.g. from an
    older transcription file) is handled as a single word.
    """
    current: list[Word] = []
    ended = False
    for segment in segments:
        words = segment.words or [Word(segment.start, segment.end, " " + segment.text)]
        for word in words:
            text = word.text.strip()
            if not text:
                continue
            punctuation_only = not any(char.isalnum() for char in text)
            if current and not punctuation_only and (
                ended or word.start - current[-1].end > max_pause
            ):
                yield current
                current = []
            current.append(word)
            ends_sentence = text.rstrip(CLOSING_CHARS).endswith(SENTENCE_END)
            ended = (ended or ends_sentence) if punctuation_only else ends_sentence
    if current:
        yield current


def assign_speakers(
    segments: Iterable[Segment], turns: Iterable[Turn], max_pause: float = 1.0
) -> list[Utterance]:
    """Return one utterance per sentence, with the raw speaker id."""
    index = SpeakerIndex(turns)
    utterances = []
    for words in split_sentences(segments, max_pause):
        weights: dict[str, float] = defaultdict(float)
        for word in words:
            for speaker, weight in index.speaker_weights(word.start, word.end).items():
                weights[speaker] += weight
        speaker = max(weights, key=weights.__getitem__) if weights else UNKNOWN_SPEAKER
        text = "".join(word.text for word in words).strip()
        utterances.append(Utterance(words[0].start, words[-1].end, speaker, text))
    return utterances


def group_consecutive(utterances: Iterable[Utterance]) -> list[Utterance]:
    """Merge consecutive utterances of the same speaker into one."""
    grouped: list[Utterance] = []
    for utterance in utterances:
        if grouped and grouped[-1].speaker == utterance.speaker:
            last = grouped[-1]
            grouped[-1] = Utterance(
                last.start, utterance.end, last.speaker, f"{last.text} {utterance.text}"
            )
        else:
            grouped.append(utterance)
    return grouped


def rename_speakers(utterances: Iterable[Utterance], label: str = "Speaker") -> list[Utterance]:
    """Replace raw ids (SPEAKER_00...) with "<label> 1", "<label> 2"...

    Speakers are numbered in order of first appearance.
    """
    names = {UNKNOWN_SPEAKER: UNKNOWN_SPEAKER}
    renamed = []
    for utterance in utterances:
        if utterance.speaker not in names:
            names[utterance.speaker] = f"{label} {len(names)}"
        renamed.append(
            Utterance(utterance.start, utterance.end, names[utterance.speaker], utterance.text)
        )
    return renamed
