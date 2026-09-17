"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

from echoes import __version__
from echoes.diarization import DEFAULT_PIPELINE
from echoes.export import FORMATS
from echoes.summary import PROVIDERS, SummaryError
from echoes.transcription import DEFAULT_MODEL

log = logging.getLogger("echoes")

# Harmless third-party warnings printed on every run
NOISY_LOGGERS = ("torch.utils.flop_counter",)
NOISY_WARNINGS = (
    r"TensorFloat-32 \(TF32\) has been disabled",
    r"std\(\): degrees of freedom is <= 0",
)

ENV_FILE = Path(".env")
ENV_FILE_HELP = (
    "file with API keys and tokens, one NAME=value per line "
    "(default: .env in the current folder, if it exists)"
)
SUMMARY_NOTE = (
    "Except with ollama, which runs locally, the transcript text is sent to the "
    "provider. API keys are read from the environment or the .env file: "
    + ", ".join(p.key_vars[0] for p in PROVIDERS.values() if p.key_vars)
    + "."
)
PROVIDERS_LIST = ", ".join(PROVIDERS)
MODEL_HELP = (
    "LLM model (default: "
    + ", ".join(f"{p.default_model} for {p.name}" for p in PROVIDERS.values() if p.default_model)
    + "; required for ollama)"
)
BASE_URL_HELP = (
    "API address, for an OpenAI-compatible service with the openai provider "
    "(OpenRouter, Groq, LM Studio...) or a remote Ollama server"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="echoes",
        description=(
            "Transcribe an audio file with speaker labels and timestamps "
            "(faster-whisper + pyannote.audio), and optionally summarize it with an LLM. "
            "Each run is saved in its own timestamped directory."
        ),
        epilog=(
            "To summarize an existing run, e.g. after replacing speaker labels by names "
            "in transcript.txt: echoes summarize RUN_DIR --provider PROVIDER "
            "(see echoes summarize --help)"
        ),
    )
    parser.add_argument("audio", type=Path, help="audio or video file (mp3, wav, m4a, mp4...)")
    parser.add_argument(
        "-o", "--output-dir", type=Path,
        help="where run directories are created (default: an echoes_runs folder next to the audio file)",
    )
    parser.add_argument(
        "--formats", type=parse_formats, default=FORMATS,
        help=f"comma-separated transcript formats among {', '.join(FORMATS)} (default: all)",
    )
    parser.add_argument(
        "--speaker-label", default="Speaker",
        help='prefix of speaker names, e.g. "Locuteur" gives "Locuteur 1" (default: Speaker)',
    )
    parser.add_argument(
        "--env-file", type=Path, default=ENV_FILE, metavar="FILE", help=ENV_FILE_HELP
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug messages")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    stt = parser.add_argument_group("transcription")
    stt.add_argument(
        "-m", "--model", default=DEFAULT_MODEL,
        help=f"Whisper model: tiny, base, small, medium, large-v3, large-v3-turbo... (default: {DEFAULT_MODEL})",
    )
    stt.add_argument(
        "-l", "--language",
        help="language code such as fr or en (default: detected from the first 30 seconds)",
    )
    stt.add_argument(
        "--hotwords",
        help='names and jargon to help recognition, e.g. "Dupont, CSTB, PLUi"',
    )
    stt.add_argument("--beam-size", type=positive_int, default=5, help="beam size (default: 5)")
    stt.add_argument(
        "--batch-size", type=positive_int, default=1,
        help="transcribe N chunks in parallel: faster on GPU but uses more memory (default: 1)",
    )
    stt.add_argument(
        "--no-vad", dest="vad", action="store_false",
        help="disable the voice activity filter that skips silences",
    )
    stt.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto",
        help="compute device (default: cuda if available)",
    )
    stt.add_argument(
        "--compute-type", default="auto",
        help="CTranslate2 compute type: float16, int8, int8_float16... "
        "(default: float16 if the GPU supports it, int8 otherwise)",
    )
    stt.add_argument(
        "--transcription", type=Path, metavar="FILE",
        help="skip Whisper and reuse the transcription.json of a previous run",
    )

    dia = parser.add_argument_group("diarization")
    dia.add_argument(
        "-n", "--num-speakers", type=positive_int, help="exact number of speakers, if known"
    )
    dia.add_argument("--min-speakers", type=positive_int, help="minimum number of speakers")
    dia.add_argument("--max-speakers", type=positive_int, help="maximum number of speakers")
    dia.add_argument(
        "--diarization-model", default=DEFAULT_PIPELINE,
        help=f"pyannote pipeline (default: {DEFAULT_PIPELINE})",
    )
    dia.add_argument(
        "--hf-token",
        help="Hugging Face token (default: HF_TOKEN from the environment or the .env file, "
        "or the token saved by `hf auth login`)",
    )

    summary = parser.add_argument_group("summary (optional)", SUMMARY_NOTE)
    summary.add_argument(
        "--summary", choices=PROVIDERS, metavar="PROVIDER",
        help=f"summarize the conversation into summary.md with an LLM: {PROVIDERS_LIST}",
    )
    summary.add_argument("--summary-model", metavar="MODEL", help=MODEL_HELP)
    summary.add_argument("--summary-base-url", metavar="URL", help=BASE_URL_HELP)
    return parser


def build_summarize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="echoes summarize",
        description=(
            "Summarize the transcript of an existing run with an LLM. transcript.txt is "
            "used, including your edits (e.g. real names instead of speaker labels). "
            "The summary is saved in the run directory, without overwriting previous ones. "
            + SUMMARY_NOTE
        ),
    )
    parser.add_argument(
        "run_dir", type=Path, metavar="RUN_DIR", help="run directory created by echoes"
    )
    parser.add_argument(
        "-p", "--provider", choices=PROVIDERS, required=True, metavar="PROVIDER",
        help=f"LLM provider: {PROVIDERS_LIST}",
    )
    parser.add_argument("-m", "--model", help=MODEL_HELP)
    parser.add_argument("--base-url", metavar="URL", help=BASE_URL_HELP)
    parser.add_argument(
        "--env-file", type=Path, default=ENV_FILE, metavar="FILE", help=ENV_FILE_HELP
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug messages")
    return parser


def parse_formats(value: str) -> tuple[str, ...]:
    formats = tuple(dict.fromkeys(f.strip().lower() for f in value.split(",") if f.strip()))
    unknown = set(formats) - set(FORMATS)
    if not formats or unknown:
        raise argparse.ArgumentTypeError(
            f"invalid format(s): {value!r} (choose among {', '.join(FORMATS)})"
        )
    return formats


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def load_env_file(path: Path) -> None:
    """Load NAME=value lines into the environment, which keeps precedence."""
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        name, sep, value = line.partition("=")
        name = name.strip().removeprefix("export ").strip()
        if not sep or not name or name.startswith("#"):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if value:
            os.environ.setdefault(name, value)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
    for message in NOISY_WARNINGS:
        warnings.filterwarnings("ignore", message=message)
    # Windows without developer mode: Hugging Face warns about missing symlinks
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["summarize"]:
        return summarize_main(argv[1:])

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size > 1 and not args.vad:
        parser.error("--batch-size relies on voice activity detection: drop --no-vad")
    if args.num_speakers and (args.min_speakers or args.max_speakers):
        parser.error("--num-speakers cannot be combined with --min/--max-speakers")
    if args.min_speakers and args.max_speakers and args.min_speakers > args.max_speakers:
        parser.error("--min-speakers is greater than --max-speakers")
    if (args.summary_model or args.summary_base_url) and not args.summary:
        parser.error("--summary-model and --summary-base-url need --summary PROVIDER")
    read_env_file(parser, args.env_file)
    setup_logging(args.verbose)

    from echoes.runner import Options, run

    options = Options(
        audio=args.audio,
        output_dir=args.output_dir,
        model=args.model,
        language=args.language,
        hotwords=args.hotwords,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        vad=args.vad,
        device=args.device,
        compute_type=args.compute_type,
        diarization_model=args.diarization_model,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        hf_token=args.hf_token,
        transcription=args.transcription,
        speaker_label=args.speaker_label,
        formats=args.formats,
        summary=args.summary,
        summary_model=args.summary_model,
        summary_base_url=args.summary_base_url,
    )
    return report(lambda: run(options), args.verbose)


def summarize_main(argv: Sequence[str]) -> int:
    parser = build_summarize_parser()
    args = parser.parse_args(argv)
    read_env_file(parser, args.env_file)
    setup_logging(args.verbose)

    from echoes.runner import summarize_run

    return report(
        lambda: summarize_run(args.run_dir, args.provider, args.model, args.base_url),
        args.verbose,
    )


def read_env_file(parser: argparse.ArgumentParser, path: Path) -> None:
    # PowerShell and cmd don't expand "~"
    path = path.expanduser()
    if path.is_file():
        load_env_file(path)
    elif path != ENV_FILE:
        parser.error(f"--env-file: file not found: {path}")


def report(action: Callable[[], Path], verbose: bool) -> int:
    """Run `action` and turn the expected errors into messages and exit codes."""
    from echoes.diarization import ModelAccessError

    try:
        path = action()
    except (FileNotFoundError, ValueError, ModelAccessError, SummaryError) as exc:
        log.error("Error: %s", exc, exc_info=verbose)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted")
        return 130
    log.info("Done: %s", path)
    return 0
