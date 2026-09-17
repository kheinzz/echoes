# diarise

Transcribe interviews, meetings and podcasts **with speaker labels and timestamps**, entirely on your own machine, using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [pyannote.audio](https://github.com/pyannote/pyannote-audio).

```text
[00:00:00 --> 00:00:12] Speaker 1: Thanks for joining us. Could you introduce yourself?

[00:00:12 --> 00:00:31] Speaker 2: Sure. I've been working on sustainable building materials for about ten years, mostly with local authorities.
```

## Features

- **Local processing**: audio never leaves your computer. Uses an NVIDIA GPU when available, CPU otherwise.
- **Accurate transcription** with Whisper (`large-v3-turbo` by default) and word-level timestamps.
- **Speaker diarization** with pyannote's `speaker-diarization-community-1` pipeline.
- **Sentence-level speaker attribution**: speaker changes inside a Whisper segment are caught, and sentences are not cut in half by imprecise turn boundaries.
- **One timestamped folder per run**: nothing gets overwritten, and each run is documented (options, versions, timings) in `run.json`.
- **Several outputs**: readable text, SRT subtitles and JSON.
- **Reuse a transcription** to re-run only the diarization, for instance with a known number of speakers.
- **Hotwords** to help Whisper with names, acronyms and jargon.
- **Works on older GPUs**: falls back to int8 when the card doesn't support float16 (e.g. Pascal generation).

## Requirements

- Python 3.10 or newer
- An NVIDIA GPU with CUDA 12 is strongly recommended. CPU works, but is much slower.
- A free [Hugging Face](https://huggingface.co/join) account, to download the pyannote model
- No FFmpeg installation needed: audio is decoded with PyAV, which ships with faster-whisper.

## Installation

```bash
git clone https://github.com/kheinzz/diarise.git
cd diarise

# with conda (or use python -m venv)
conda create -n diarise python=3.11
conda activate diarise

# 1. PyTorch: pick the command matching your system on https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 2. diarise and its dependencies
pip install -e .
```

### Hugging Face access (once)

The pyannote models are free but gated:

1. Accept the conditions on [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
2. Create a **read** token on [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). For a *fine-grained* token, tick **"Read access to contents of all public gated repos you can access"**.
3. Save the token on your machine:

   ```bash
   hf auth login
   ```

   Alternatively, set the `HF_TOKEN` environment variable or pass `--hf-token`. Never write the token in the code.

## Usage

```bash
# simplest form: language is detected automatically
diarise interview.mp3

# French interview, French speaker labels, names Whisper should know
diarise interview.mp3 --language fr --speaker-label Locuteur --hotwords "Dupont, CSTB, PLUi"

# the number of speakers is known
diarise interview.mp3 --num-speakers 3

# re-run only the diarization, reusing the transcription of a previous run
diarise interview.mp3 --num-speakers 3 \
    --transcription diarise_runs/interview_2026-09-17_15-07-00/transcription.json
```

`python -m diarise` works too. Run `diarise --help` for all options.

### Main options

| Option | Default | Description |
|---|---|---|
| `-o`, `--output-dir` | `diarise_runs/` next to the audio | Where run folders are created |
| `-m`, `--model` | `large-v3-turbo` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`... |
| `-l`, `--language` | auto-detected | Language code (`fr`, `en`...) |
| `--hotwords` | | Names and jargon to help recognition |
| `-n`, `--num-speakers` | auto | Exact number of speakers |
| `--min-speakers`, `--max-speakers` | auto | Bounds on the number of speakers |
| `--speaker-label` | `Speaker` | Prefix of speaker names (`Speaker 1`, `Speaker 2`...) |
| `--formats` | `txt,srt,json` | Transcript formats to write |
| `--transcription` | | Reuse a `transcription.json` and skip Whisper |
| `--batch-size` | `1` | Transcribe N chunks in parallel (faster, uses more GPU memory) |
| `--device` | `auto` | `cuda` or `cpu` |
| `--compute-type` | `auto` | `float16` if the GPU supports it, `int8` otherwise |
| `--no-vad` | | Disable the filter that skips silences |

### Python API

```python
from pathlib import Path
from diarise.runner import Options, run

run_dir = run(Options(audio=Path("interview.mp3"), language="fr", num_speakers=2))
```

## Output

Each run creates its own folder, by default in a `diarise_runs` folder next to the audio file:

```text
diarise_runs/
└── interview_2026-09-17_15-07-00/
    ├── transcript.txt       readable transcript, one paragraph per speaker turn
    ├── transcript.srt       subtitles, one cue per sentence
    ├── transcript.json      speaker turns, for further processing
    ├── transcription.json   raw Whisper output with word timestamps (reusable)
    ├── diarization.json     raw speaker turns from pyannote
    └── run.json             options, package versions, timings and status
```

If a run fails or is interrupted, `run.json` records the error. Files from the steps that completed are kept, so a finished transcription can be reused with `--transcription`.

## How it works

1. The audio is decoded once to 16 kHz mono.
2. **Transcription**: Whisper produces text segments with word-level timestamps. Silences are skipped by a voice activity filter, which also limits hallucinations.
3. **Diarization**: pyannote finds who speaks when. The *exclusive* variant is used, with at most one speaker at a time.
4. **Alignment**: words are grouped into sentences, split on punctuation or on pauses longer than one second. Each sentence goes to the speaker who talks the most during its words, or to the nearest speaker when nothing overlaps. Consecutive sentences from the same speaker are then merged into turns.

## Tips

- **GPU memory**: `large-v3` needs noticeably more memory than `large-v3-turbo`. On a 5 GB card (Quadro P2200), `large-v3` did not fit while `large-v3-turbo` runs fine. Closing GPU-hungry applications (browsers, video calls, GIS software) also helps.
- **Too many speakers?** A short noise or a laugh can create a spurious speaker. Re-run with `--num-speakers` or `--max-speakers`, reusing the transcription to save time.
- **Wrong language?** Detection only listens to the first 30 seconds. Set `--language` if the recording starts with silence or music.
- **Misspelled names?** Add them to `--hotwords`.

## Privacy

- All processing happens locally. Models are downloaded once from Hugging Face and cached.
- pyannote.audio 4 sends anonymous usage metrics (such as file duration and number of speakers) by default. **diarise turns them off**, unless you explicitly set `PYANNOTE_METRICS_ENABLED=1`.
- Recordings and transcripts often contain personal data. The `.gitignore` excludes audio files and run folders, but handle them according to your local regulations (e.g. GDPR).

## Development

```bash
pip install -e ".[dev]"
pytest
```

The unit tests cover the alignment and export logic. They don't need a GPU or any model.

## License

[MIT](LICENSE). The models have their own terms: see the model cards of [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo) and [pyannote speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [OpenAI Whisper](https://github.com/openai/whisper) and [pyannote.audio](https://github.com/pyannote/pyannote-audio).
