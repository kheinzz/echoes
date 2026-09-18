"""Speech-to-text with faster-whisper."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from echoes.schema import Segment, Word

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL = "large-v3-turbo"


@dataclass
class TranscriptionResult:
    segments: list[Segment]
    language: str
    language_probability: float


def pick_compute_type(device: str) -> str:
    """Return the fastest compute type supported by `device`.

    float16 is preferred on GPU, but older cards (Pascal generation and
    earlier) can't run it efficiently: fall back to int8.
    """
    import ctranslate2

    if device == "cuda" and "float16" in ctranslate2.get_supported_compute_types("cuda"):
        return "float16"
    return "int8"


def transcribe(
    audio: np.ndarray,
    *,
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    beam_size: int = 5,
    batch_size: int = 1,
    vad: bool = True,
    hotwords: str | None = None,
) -> TranscriptionResult:
    """Transcribe a 16 kHz mono waveform, with word-level timestamps."""
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    log.info("Loading Whisper model '%s' on %s (%s)", model_name, device, compute_type)
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    options = dict(
        language=language,
        beam_size=beam_size,
        vad_filter=vad,
        hotwords=hotwords,
        word_timestamps=True,
        log_progress=True,
    )
    if batch_size > 1:
        segments, info = BatchedInferencePipeline(model).transcribe(
            audio, batch_size=batch_size, **options
        )
    else:
        segments, info = model.transcribe(audio, **options)

    if language is None:
        log.info(
            "Detected language: %s (probability %.2f)", info.language, info.language_probability
        )
    # `segments` is a lazy generator: the actual transcription happens here
    result = [
        Segment(
            start=segment.start,
            end=segment.end,
            text=segment.text.strip(),
            words=[Word(word.start, word.end, word.word) for word in segment.words or []],
        )
        for segment in segments
    ]
    return TranscriptionResult(result, info.language, info.language_probability)
