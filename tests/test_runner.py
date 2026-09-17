import argparse
import json

import pytest
from conftest import gemini_answer

from echoes.cli import parse_formats
from echoes.runner import (
    Options,
    create_run_dir,
    format_duration,
    load_transcription,
    read_transcript,
    run_language,
    summarize_command,
    summarize_run,
    write_new_file,
)


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


def test_summary_key_is_not_saved_in_the_manifest(tmp_path):
    options = Options(audio=tmp_path / "a.mp3", summary="gemini", summary_api_key="sk-secret")
    assert "sk-secret" not in json.dumps(options.to_json())
    assert "sk-secret" not in repr(options)
    assert options.to_json()["summary"] == "gemini"


def test_write_new_file_never_overwrites(tmp_path):
    first = write_new_file(tmp_path / "summary.md", "first")
    second = write_new_file(tmp_path / "summary.md", "second")
    assert (first.name, second.name) == ("summary.md", "summary_2.md")
    assert first.read_text(encoding="utf-8") == "first"


def test_read_transcript(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_transcript(tmp_path)
    utterance = {"start": 0.0, "end": 4.0, "speaker": "Speaker 1", "text": "Hello."}
    (tmp_path / "transcript.json").write_text(
        json.dumps({"speakers": ["Speaker 1"], "utterances": [utterance]}), encoding="utf-8"
    )
    assert read_transcript(tmp_path) == "[00:00:00 --> 00:00:04] Speaker 1: Hello.\n\n"
    # transcript.txt wins, as the user may have edited it
    (tmp_path / "transcript.txt").write_text("Alice: Hello.", encoding="utf-8")
    assert read_transcript(tmp_path) == "Alice: Hello."


def test_run_language(tmp_path):
    assert run_language(tmp_path) is None
    (tmp_path / "transcription.json").write_text(
        json.dumps({"language": "fr", "segments": []}), encoding="utf-8"
    )
    assert run_language(tmp_path) == "fr"
    (tmp_path / "run.json").write_text(json.dumps({"options": {"language": "en"}}), encoding="utf-8")
    assert run_language(tmp_path) == "en"


def test_summarize_run(tmp_path, fake_llm):
    (tmp_path / "transcript.txt").write_text("Alice: I plan cities.", encoding="utf-8")
    (tmp_path / "run.json").write_text(json.dumps({"options": {"language": "fr"}}), encoding="utf-8")
    fake_llm.reply(gemini_answer("# Résumé"), gemini_answer("# Résumé 2"))

    first = summarize_run(tmp_path, "gemini", api_key="k")
    second = summarize_run(tmp_path, "gemini", api_key="k")

    assert (first.name, second.name) == ("summary.md", "summary_2.md")
    text = first.read_text(encoding="utf-8")
    assert text.startswith("---\nprovider: gemini\nmodel: gemini-test\n")
    assert text.endswith("# Résumé\n")
    body = json.dumps(fake_llm.body(0), ensure_ascii=False)
    assert "Alice: I plan cities." in body
    assert '\\"fr\\"' in body


def test_retry_command_repeats_the_summary_options(tmp_path):
    options = Options(audio=tmp_path / "a.mp3", summary="ollama", summary_model="qwen3")
    assert summarize_command(tmp_path, options) == (
        f'echoes summarize "{tmp_path}" --provider ollama --model "qwen3"'
    )
