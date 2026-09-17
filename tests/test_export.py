import json

import pytest

from diarise.export import format_timestamp, render
from diarise.schema import Utterance

SENTENCES = [
    Utterance(0.0, 1.5, "Speaker 1", "Hello."),
    Utterance(1.5, 3.25, "Speaker 1", "How are you?"),
    Utterance(3726.5, 3728.0, "Speaker 2", "Fine, thanks."),
]
TURNS = [
    Utterance(0.0, 3.25, "Speaker 1", "Hello. How are you?"),
    Utterance(3726.5, 3728.0, "Speaker 2", "Fine, thanks."),
]


@pytest.mark.parametrize(
    ("seconds", "millis", "expected"),
    [
        (0, False, "00:00:00"),
        (59.4, False, "00:00:59"),
        (3726.5, False, "01:02:06"),
        (3726.5, True, "01:02:06,500"),
        (1.9996, True, "00:00:02,000"),
    ],
)
def test_format_timestamp(seconds, millis, expected):
    assert format_timestamp(seconds, millis) == expected


def test_txt_uses_speaker_turns():
    assert render("txt", SENTENCES, TURNS) == (
        "[00:00:00 --> 00:00:03] Speaker 1: Hello. How are you?\n\n"
        "[01:02:06 --> 01:02:08] Speaker 2: Fine, thanks.\n\n"
    )


def test_srt_uses_one_cue_per_sentence():
    assert render("srt", SENTENCES, TURNS) == (
        "1\n00:00:00,000 --> 00:00:01,500\n[Speaker 1] Hello.\n\n"
        "2\n00:00:01,500 --> 00:00:03,250\n[Speaker 1] How are you?\n\n"
        "3\n01:02:06,500 --> 01:02:08,000\n[Speaker 2] Fine, thanks.\n"
    )


def test_json_lists_speakers_and_turns():
    data = json.loads(render("json", SENTENCES, TURNS))
    assert data["speakers"] == ["Speaker 1", "Speaker 2"]
    assert data["utterances"][0] == {
        "start": 0.0, "end": 3.25, "speaker": "Speaker 1", "text": "Hello. How are you?"
    }


def test_unknown_format():
    with pytest.raises(ValueError):
        render("docx", SENTENCES, TURNS)
