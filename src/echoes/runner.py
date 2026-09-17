"""Pipeline orchestration: each run gets its own timestamped directory.

    <output_dir>/<audio name>_<YYYY-MM-DD_HH-MM-SS>/
        run.json             options, versions, timings and status
        transcription.json   Whisper segments with word timestamps
        diarization.json     speaker turns
        transcript.txt|srt|json
        summary.md           optional summary written by an LLM
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
from echoes.export import FORMATS, render, to_txt
from echoes.merge import assign_speakers, group_consecutive, rename_speakers
from echoes.schema import Segment, Turn, Utterance, to_dicts
from echoes.summary import Client, Summary, SummaryError, make_client, summarize
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
    # LLM provider for the optional summary (see echoes.summary.PROVIDERS)
    summary: str | None = None
    summary_model: str | None = None
    summary_base_url: str | None = None
    summary_api_key: str | None = field(default=None, repr=False)

    def to_json(self) -> dict:
        data = asdict(self)
        del data["hf_token"], data["summary_api_key"]
        return {key: str(value) if isinstance(value, Path) else value for key, value in data.items()}


class Manifest:
    """run.json, rewritten after each step so a failed run stays documented."""

    def __init__(self, path: Path, total_steps: int, **data):
        self.path = path
        self.total_steps = total_steps
        self.data = {**data, "status": "running", "steps": {}}
        self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    @contextmanager
    def step(self, number: int, name: str) -> Iterator[dict]:
        log.info("Step %d/%d - %s", number, self.total_steps, name)
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
    client = (
        make_client(opts.summary, opts.summary_model, opts.summary_base_url, opts.summary_api_key)
        if opts.summary
        else None
    )

    run_dir = create_run_dir(opts.output_dir or audio_path.parent / RUNS_DIRNAME, audio_path.stem)
    log.info("Run directory: %s", run_dir)
    device = resolve_device(opts.device)
    log.info("Compute device: %s", device)
    manifest = Manifest(
        run_dir / "run.json",
        total_steps=4 if client else 3,
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
        _run_steps(opts, audio_path, transcription_path, run_dir, device, manifest, client)
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
    client: Client | None,
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

    if client is None:
        return
    with manifest.step(4, "summary") as info:
        info["provider"] = client.provider.name
        if not grouped:
            log.warning("The transcript is empty: no summary")
            info["skipped"] = "empty transcript"
            return
        language = opts.language or transcription.get("language")
        try:
            path, summary = save_summary(run_dir, to_txt(grouped), language, client)
        except SummaryError as exc:
            raise SummaryError(
                f"{exc}\nThe transcript is saved. To retry only the summary, run:\n"
                f"  {summarize_command(run_dir, opts)}"
            ) from exc
        info.update(model=summary.model, file=path.name)


def summarize_run(
    run_dir: Path,
    provider: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Path:
    """Summarize the transcript of an existing run, and return the summary file.

    transcript.txt is preferred, so that manual edits (e.g. real names instead
    of speaker labels) are taken into account.
    """
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    client = make_client(provider, model, base_url, api_key)
    path, _ = save_summary(run_dir, read_transcript(run_dir), run_language(run_dir), client)
    return path


def save_summary(
    run_dir: Path, transcript: str, language: str | None, client: Client
) -> tuple[Path, Summary]:
    summary = summarize(transcript, client, language=language)
    path = write_new_file(run_dir / "summary.md", summary.to_markdown())
    log.info("Wrote %s", path)
    return path, summary


def summarize_command(run_dir: Path, opts: Options) -> str:
    command = f'echoes summarize "{run_dir}" --provider {opts.summary}'
    if opts.summary_model:
        command += f' --model "{opts.summary_model}"'
    if opts.summary_base_url:
        command += f' --base-url "{opts.summary_base_url}"'
    return command


def read_transcript(run_dir: Path) -> str:
    """Transcript text of a previous run."""
    path = run_dir / "transcript.txt"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    path = run_dir / "transcript.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return to_txt([Utterance(**utterance) for utterance in data["utterances"]])
    raise FileNotFoundError(f"No transcript.txt or transcript.json in {run_dir}")


def run_language(run_dir: Path) -> str | None:
    """Language of a previous run: set by the user, or detected by Whisper."""
    path = run_dir / "run.json"
    if path.is_file():
        options = json.loads(path.read_text(encoding="utf-8")).get("options", {})
        if options.get("language"):
            return options["language"]
    path = run_dir / "transcription.json"
    if path.is_file():
        return load_transcription(path).get("language")
    return None


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


def write_new_file(path: Path, text: str) -> Path:
    """Write `text` to `path`, or to `<stem>_2<suffix>`... if it already exists."""
    candidate, suffix = path, 1
    while True:
        try:
            with candidate.open("x", encoding="utf-8") as file:
                file.write(text)
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = path.with_name(f"{path.stem}_{suffix}{path.suffix}")


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
