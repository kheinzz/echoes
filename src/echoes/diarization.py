"""Speaker diarization (who speaks when) with pyannote.audio."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from echoes.schema import Turn

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

DEFAULT_PIPELINE = "pyannote/speaker-diarization-community-1"


class ModelAccessError(RuntimeError):
    """The diarization pipeline could not be downloaded from Hugging Face."""


def diarize(
    audio: np.ndarray,
    *,
    sample_rate: int,
    pipeline_name: str = DEFAULT_PIPELINE,
    device: str = "cpu",
    token: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[Turn]:
    """Return non-overlapping speaker turns for a mono waveform."""
    log.info("Loading diarization pipeline '%s' on %s", pipeline_name, device)
    # pyannote.audio sends usage metrics by default: opt out, unless the user
    # explicitly set this variable
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")

    import torch
    from huggingface_hub.errors import HfHubHTTPError
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    try:
        # token=None lets huggingface_hub use $HF_TOKEN or the `hf auth login` token
        pipeline = Pipeline.from_pretrained(pipeline_name, token=token)
    except HfHubHTTPError as exc:
        raise ModelAccessError(access_help(pipeline_name)) from exc
    if pipeline is None:
        raise ModelAccessError(access_help(pipeline_name))
    pipeline.to(torch.device(device))

    # Passing the waveform in memory bypasses pyannote's own audio decoding
    # (torchcodec), which requires FFmpeg shared libraries on Windows
    waveform = torch.from_numpy(audio).unsqueeze(0)
    with ProgressHook() as hook:
        output = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hook=hook,
        )

    # The "exclusive" variant has at most one speaker at a time, which is what
    # we need to align speakers with the transcript timestamps
    return [
        Turn(start=turn.start, end=turn.end, speaker=speaker)
        for turn, _, speaker in output.exclusive_speaker_diarization.itertracks(yield_label=True)
    ]


def access_help(pipeline_name: str) -> str:
    return (
        f"Could not download '{pipeline_name}' from Hugging Face.\n"
        "Check the model name and your connection. pyannote models are gated:\n"
        f"  1. accept the conditions at https://huggingface.co/{pipeline_name}\n"
        "  2. provide a read token with --hf-token, the HF_TOKEN environment variable\n"
        "     or `hf auth login` (fine-grained tokens also need the permission\n"
        '     "Read access to contents of all public gated repos you can access")'
    )
