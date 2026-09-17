"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from collections.abc import Sequence
from pathlib import Path

from diarise import __version__
from diarise.diarization import DEFAULT_PIPELINE
from diarise.export import FORMATS
from diarise.transcription import DEFAULT_MODEL

log = logging.getLogger("diarise")

# Harmless third-party warnings printed on every run
NOISY_LOGGERS = ("torch.utils.flop_counter",)
NOISY_WARNINGS = (
    r"TensorFloat-32 \(TF32\) has been disabled",
    r"std\(\): degrees of freedom is <= 0",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diarise",
        description=(
            "Transcribe an audio file with speaker labels and timestamps "
            "(faster-whisper + pyannote.audio). Each run is saved in its own "
            "timestamped directory."
        ),
    )
    parser.add_argument("audio", type=Path, help="audio or video file (mp3, wav, m4a, mp4...)")
    parser.add_argument(
        "-o", "--output-dir", type=Path,
        help="where run directories are created (default: a diarise_runs folder next to the audio file)",
    )
    parser.add_argument(
        "--formats", type=parse_formats, default=FORMATS,
        help=f"comma-separated transcript formats among {', '.join(FORMATS)} (default: all)",
    )
    parser.add_argument(
        "--speaker-label", default="Speaker",
        help='prefix of speaker names, e.g. "Locuteur" gives "Locuteur 1" (default: Speaker)',
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
        help="Hugging Face token (default: $HF_TOKEN or the token saved by `hf auth login`)",
    )
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size > 1 and not args.vad:
        parser.error("--batch-size relies on voice activity detection: drop --no-vad")
    if args.num_speakers and (args.min_speakers or args.max_speakers):
        parser.error("--num-speakers cannot be combined with --min/--max-speakers")
    if args.min_speakers and args.max_speakers and args.min_speakers > args.max_speakers:
        parser.error("--min-speakers is greater than --max-speakers")
    setup_logging(args.verbose)

    from diarise.diarization import ModelAccessError
    from diarise.runner import Options, run

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
    )
    try:
        run_dir = run(options)
    except (FileNotFoundError, ValueError, ModelAccessError) as exc:
        log.error("Error: %s", exc, exc_info=args.verbose)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted")
        return 130
    log.info("Done: %s", run_dir)
    return 0
