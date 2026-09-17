import argparse
import json

import pytest

from echoes.cli import parse_formats
from echoes.runner import Options, create_run_dir, format_duration, load_transcription


def test_run_dirs_never_collide(tmp_path):
    first = create_run_dir(tmp_path / "runs", "interview")
    second = create_run_dir(tmp_path / "runs", "interview")
    assert first.is_dir() and second.is_dir()
    assert first != second
    assert first.name.startswith("interview_")


def test_load_transcription_accepts_legacy_list(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps([{"start": 0.0, "end": 1.0, "text": "Hi."}]), encoding="utf-8")
    assert load_transcription(path) == {"segments": [{"start": 0.0, "end": 1.0, "text": "Hi."}]}


def test_load_transcription_rejects_other_json(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"turns": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_transcription(path)


def test_token_is_not_saved_in_the_manifest(tmp_path):
    options = Options(audio=tmp_path / "a.mp3", hf_token="hf_secret")
    assert "hf_secret" not in json.dumps(options.to_json())
    assert "hf_secret" not in repr(options)


def test_parse_formats():
    assert parse_formats("SRT, txt,srt") == ("srt", "txt")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_formats("txt,docx")


@pytest.mark.parametrize(
    ("seconds", "expected"), [(5.4, "5s"), (65, "1m05s"), (6185.2, "1h43m05s")]
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected
