"""Pipeline orchestration: each run gets its own timestamped directory.

    <output_dir>/<audio name>_<YYYY-MM-DD_HH-MM-SS>/
        run.json             options, versions, timings and status
        transcription.json   Whisper segments with word timestamps
        diarization.json     speaker turns
        transcript.txt|srt|json
"""

from __future__ import annotations

import json
import logging
import platform
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from echoes import __version__
from echoes.diarization import DEFAULT_PIPELINE
from echoes.export import FORMATS, render
from echoes.merge import assign_speakers, group_consecutive, rename_speakers
from echoes.schema import Segment, Turn, to_dicts
from echoes.transcription import DEFAULT_MODEL

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
RUNS_DIRNAME = "echoes_runs"
TRACKED_PACKAGES = ("faster-whisper", "ctranslate2", "pyannote.audio", "torch")


@dataclass
class Options:
    audio: Path
    output_dir: Path | None = None
    model: str = DEFAULT_MODEL
    language: str | None = None
    hotwords: str | None = None
    beam_size: int = 5
    batch_size: int = 1
    vad: bool = True
    device: str = "auto"
    compute_type: str = "auto"
    diarization_model: str = DEFAULT_PIPELINE
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    hf_token: str | None = field(default=None, repr=False)
    transcription: Path | None = None
    speaker_label: str = "Speaker"
    formats: tuple[str, ...] = FORMATS

    def to_json(self) -> dict:
        data = asdict(self)
        del data["hf_token"]
        return {key: str(value) if isinstance(value, Path) else value for key, value in data.items()}


class Manifest:
    """run.json, rewritten after each step so a failed run stays documented."""

    def __init__(self, path: Path, **data):
        self.path = path
        self.data = {**data, "status": "running", "steps": {}}
        self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    @contextmanager
    def step(self, number: int, name: str) -> Iterator[dict]:
        log.info("Step %d/3 - %s", number, name)
        info: dict = {}
        self.data["steps"][name] = info
        start = time.perf_counter()
        yield info
        info["seconds"] = round(time.perf_counter() - start, 1)
        log.info("%s done in %s", name.capitalize(), format_duration(info["seconds"]))
        self.save()


def run(opts: Options) -> Path:
    """Run the whole pipeline and return the run directory."""
    audio_path = opts.audio.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    transcription_path = opts.transcription.resolve() if opts.transcription else None
    if transcription_path and not transcription_path.is_file():
        raise FileNotFoundError(f"Transcription file not found: {transcription_path}")

    run_dir = create_run_dir(opts.output_dir or audio_path.parent / RUNS_DIRNAME, audio_path.stem)
    log.info("Run directory: %s", run_dir)
    device = resolve_device(opts.device)
    log.info("Compute device: %s", device)
    manifest = Manifest(
        run_dir / "run.json",
        echoes_version=__version__,
        started_at=datetime.now().isoformat(timespec="seconds"),
        options=opts.to_json(),
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": device,
            "packages": package_versions(),
        },
    )
    try:
        _run_steps(opts, audio_path, transcription_path, run_dir, device, manifest)
    except BaseException as exc:
        manifest.data["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest.data["error"] = f"{type(exc).__name__}: {exc}"
        manifest.save()
        raise
    manifest.data["status"] = "done"
    manifest.data["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest.save()
    return run_dir


def _run_steps(
    opts: Options,
    audio_path: Path,
    transcription_path: Path | None,
    run_dir: Path,
    device: str,
    manifest: Manifest,
) -> None:
    from faster_whisper import decode_audio

    log.info("Decoding %s", audio_path.name)
    audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE
    log.info("Audio duration: %s", format_duration(duration))
    manifest.data["audio"] = {"path": str(audio_path), "duration_seconds": round(duration, 1)}

    with manifest.step(1, "transcription") as info:
        if transcription_path:
            log.info("Reusing %s", transcription_path)
            transcription = load_transcription(transcription_path)
            info["reused_from"] = str(transcription_path)
        else:
            from echoes.transcription import pick_compute_type, transcribe

            compute_type = (
                pick_compute_type(device) if opts.compute_type == "auto" else opts.compute_type
            )
            info["compute_type"] = compute_type
            result = transcribe(
                audio,
                model_name=opts.model,
                device=device,
                compute_type=compute_type,
                language=opts.language,
                beam_size=opts.beam_size,
                batch_size=opts.batch_size,
                vad=opts.vad,
                hotwords=opts.hotwords,
            )
            transcription = {
                "model": opts.model,
                "language": result.language,
                "language_probability": round(result.language_probability, 3),
                "segments": to_dicts(result.segments),
            }
        write_json(run_dir / "transcription.json", transcription)
        segments = [Segment.from_dict(segment) for segment in transcription["segments"]]
        info["segments"] = len(segments)

    with manifest.step(2, "diarization") as info:
        from echoes.diarization import diarize

        turns = diarize(
            audio,
            sample_rate=SAMPLE_RATE,
            pipeline_name=opts.diarization_model,
            device=device,
            token=opts.hf_token,
            num_speakers=opts.num_speakers,
            min_speakers=opts.min_speakers,
            max_speakers=opts.max_speakers,
        )
        speakers = sorted({turn.speaker for turn in turns})
        info["speakers"] = len(speakers)
        write_json(
            run_dir / "diarization.json",
            {"model": opts.diarization_model, "speakers": speakers, "turns": to_dicts(turns)},
        )
        log.info("Found %d speaker(s)", len(speakers))

    with manifest.step(3, "export") as info:
        sentences = rename_speakers(assign_speakers(segments, turns), opts.speaker_label)
        grouped = group_consecutive(sentences)
        info["files"] = []
        for fmt in opts.formats:
            path = run_dir / f"transcript.{fmt}"
            path.write_text(render(fmt, sentences, grouped), encoding="utf-8")
            info["files"].append(path.name)
            log.info("Wrote %s", path)


def create_run_dir(root: Path, name: str) -> Path:
    """Create `root/<name>_<timestamp>`, with a numeric suffix if it exists."""
    base = root / f"{name}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    run_dir, suffix = base, 1
    while True:
        try:
            run_dir.mkdir(parents=True)
            return run_dir
        except FileExistsError:
            suffix += 1
            run_dir = base.with_name(f"{base.name}_{suffix}")


def load_transcription(path: Path) -> dict:
    """Load a transcription.json, also accepting a bare list of segments."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"segments": data}
    if not isinstance(data.get("segments"), list):
        raise ValueError(f"{path} does not look like a transcription file")
    return data


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
