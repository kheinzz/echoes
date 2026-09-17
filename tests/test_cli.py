import os

import pytest
from conftest import gemini_answer

from echoes.cli import load_env_file, main


@pytest.fixture
def run_dir(tmp_path):
    path = tmp_path / "interview_2026-09-17_10-00-00"
    path.mkdir()
    (path / "transcript.txt").write_text(
        "[00:00:00 --> 00:00:04] Speaker 1: What is your job?\n\n", encoding="utf-8"
    )
    return path


def test_load_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ALREADY_SET", "from environment")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "﻿# comment\n"
        "HF_TOKEN=hf_abc\n"
        "export GEMINI_API_KEY = 'quoted key'\n"
        'OPENAI_API_KEY="double=quoted"\n'
        "MISTRAL_API_KEY=\n"
        "ALREADY_SET=from file\n"
        "not a variable\n",
        encoding="utf-8",
    )

    load_env_file(env_file)

    assert os.environ["HF_TOKEN"] == "hf_abc"
    assert os.environ["GEMINI_API_KEY"] == "quoted key"
    assert os.environ["OPENAI_API_KEY"] == "double=quoted"
    assert "MISTRAL_API_KEY" not in os.environ
    assert os.environ["ALREADY_SET"] == "from environment"


def test_summarize_command_reads_the_env_file(run_dir, tmp_path, fake_llm):
    env_file = tmp_path / "keys.env"
    env_file.write_text("GEMINI_API_KEY=file-key\n", encoding="utf-8")
    fake_llm.reply(gemini_answer("# Summary"))

    code = main(["summarize", str(run_dir), "-p", "gemini", "--env-file", str(env_file)])

    assert code == 0
    assert (run_dir / "summary.md").is_file()
    assert "file-key" in str(fake_llm.requests[0].header_items())


def test_summarize_command_reports_errors(run_dir, tmp_path, monkeypatch, fake_llm):
    monkeypatch.chdir(tmp_path)
    assert main(["summarize", str(run_dir), "-p", "gemini"]) == 1
    assert main(["summarize", str(tmp_path / "missing"), "-p", "ollama", "-m", "x"]) == 1
    assert not fake_llm.requests


def test_missing_key_is_detected_before_transcribing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audio = tmp_path / "interview.mp3"
    audio.write_bytes(b"")

    assert main([str(audio), "--summary", "openai"]) == 1
    assert not (tmp_path / "echoes_runs").exists()


@pytest.mark.parametrize(
    "args",
    [
        ["interview.mp3", "--summary-model", "gpt-test"],
        ["interview.mp3", "--summary", "acme"],
        ["interview.mp3", "--env-file", "missing.env"],
        ["summarize", "run_dir"],
    ],
)
def test_invalid_arguments(args, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        main(args)
    assert exit_info.value.code == 2
