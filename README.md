# echoes

[![tests](https://github.com/kheinzz/echoes/actions/workflows/tests.yml/badge.svg)](https://github.com/kheinzz/echoes/actions/workflows/tests.yml)

Transcribe interviews, meetings and podcasts **with speaker labels and timestamps**, entirely on your own machine, using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [pyannote.audio](https://github.com/pyannote/pyannote-audio). Optionally, summarize the conversation with the LLM of your choice.

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
- **Optional summary** with Gemini, OpenAI, Anthropic, Mistral, any OpenAI-compatible service, or a local model through Ollama. Interviews are summarized question by question; open discussions are summarized by topic.
- **Reuse a transcription** to re-run only the diarization, for instance with a known number of speakers.
- **Hotwords** to help Whisper with names, acronyms and jargon.
- **Works on older GPUs**: falls back to int8 when the card doesn't support float16 (e.g. Pascal generation).

## Requirements

- Python 3.10 or newer
- An NVIDIA GPU with CUDA 12 is strongly recommended. CPU works, but is much slower.
- A free [Hugging Face](https://huggingface.co/join) account, to download the pyannote model
- For summaries only: an API key from an LLM provider, or [Ollama](https://ollama.com) installed locally
- No FFmpeg installation needed: audio is decoded with PyAV, which ships with faster-whisper.

## Installation

```bash
git clone https://github.com/kheinzz/echoes.git
cd echoes

# with conda (or use python -m venv)
conda create -n echoes python=3.11
conda activate echoes

# 1. PyTorch: pick the command matching your system on https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 2. echoes and its dependencies
pip install -e .
```

### Keys and tokens: the `.env` file

echoes reads its secrets from a `.env` file, so that they never appear in your commands or in the code:

1. Copy [`.env.example`](.env.example) to a file named `.env`, **in the folder from which you run `echoes`**:

   ```bash
   cp .env.example .env        # Windows: copy .env.example .env
   ```

2. Fill in the values you need:

   ```ini
   # Hugging Face read token, for the diarization model (required)
   HF_TOKEN=hf_xxxxxxxxxxxxxxxx

   # Key of the LLM provider used for summaries (optional)
   GEMINI_API_KEY=xxxxxxxxxxxxxxxx
   ```

- To keep the file elsewhere, pass its path with `--env-file`, e.g. `--env-file ~/.config/echoes.env`.
- Variables already set in the environment take precedence over the file, so `HF_TOKEN=... echoes ...` or a system-wide variable also works.
- `.env` is ignored by git. Never commit it, and never write a key in the code. If a key leaks, revoke it and create a new one.

### Hugging Face access (once)

The pyannote models are free but gated:

1. Accept the conditions on [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
2. Create a **read** token on [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). For a *fine-grained* token, tick **"Read access to contents of all public gated repos you can access"**.
3. Put the token in your `.env` file as `HF_TOKEN=...`. Alternatively, run `hf auth login` once, or pass `--hf-token`.

## Usage

```bash
# simplest form: language is detected automatically
echoes interview.mp3

# French interview, French speaker labels, names Whisper should know
echoes interview.mp3 --language fr --speaker-label Locuteur --hotwords "Dupont, CSTB, PLUi"

# the number of speakers is known
echoes interview.mp3 --num-speakers 3

# transcribe, then summarize with Gemini
echoes interview.mp3 --language fr --summary gemini

# re-run only the diarization, reusing the transcription of a previous run
echoes interview.mp3 --num-speakers 3 \
    --transcription echoes_runs/interview_2026-09-17_15-07-00/transcription.json

# summarize an existing run (see "Summary" below)
echoes summarize echoes_runs/interview_2026-09-17_15-07-00 --provider gemini
```

`python -m echoes` works too. Run `echoes --help` and `echoes summarize --help` for all options.

### Main options

| Option | Default | Description |
|---|---|---|
| `-o`, `--output-dir` | `echoes_runs/` next to the audio | Where run folders are created |
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
| `--summary` | no summary | Summarize with `gemini`, `openai`, `anthropic`, `mistral` or `ollama` |
| `--summary-model` | see below | LLM model |
| `--summary-base-url` | provider's API | Address of an OpenAI-compatible service or of a remote Ollama server |
| `--env-file` | `.env` | File with the keys and tokens |

### Python API

```python
from pathlib import Path
from echoes.runner import Options, run, summarize_run

# the keys are read from the environment (the .env file is only loaded by the CLI),
# or passed with hf_token=... and summary_api_key=...
run_dir = run(Options(audio=Path("interview.mp3"), language="fr", num_speakers=2, summary="gemini"))

# summarize an existing run
summary_file = summarize_run(run_dir, "ollama", model="qwen3")
```

## Summary (optional)

With `--summary PROVIDER`, a fourth step sends the transcript to an LLM and writes `summary.md` in the run folder. The model first decides what kind of conversation it is:

- **interview**: one section per question, with its timestamp and a summary of the answer, followed by the key points;
- **open discussion**: a summary by topic, followed by the decisions and next steps, if any.

The summary is written in the language of the transcript.

### Providers

| `--summary` | Default model | Key in `.env` | Get a key |
|---|---|---|---|
| `gemini` | `gemini-flash-latest` | `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/apikey) (free tier available) |
| `openai` | `gpt-5-mini` | `OPENAI_API_KEY` | [OpenAI platform](https://platform.openai.com/api-keys) |
| `anthropic` | `claude-sonnet-5` | `ANTHROPIC_API_KEY` | [Anthropic console](https://console.anthropic.com/settings/keys) |
| `mistral` | `mistral-medium-latest` | `MISTRAL_API_KEY` | [Mistral console](https://console.mistral.ai/api-keys) |
| `ollama` | none: set `--summary-model` | no key | runs locally, see below |

- **Another model**: `--summary-model`, e.g. `--summary gemini --summary-model gemini-pro-latest`.
- **OpenAI-compatible services** (OpenRouter, Groq, LM Studio, vLLM...): use `--summary openai` with `--summary-base-url`, and put that service's key in `OPENAI_API_KEY`. A local server that needs no key works too.

  ```bash
  echoes interview.mp3 --summary openai --summary-base-url https://openrouter.ai/api/v1 --summary-model MODEL_NAME
  ```

- **Ollama (fully local)**: install [Ollama](https://ollama.com), download a model (`ollama pull qwen3`), then:

  ```bash
  echoes interview.mp3 --summary ollama --summary-model qwen3
  ```

  echoes asks Ollama for a context window large enough for the whole transcript, since the default one is often too small. Large models need a lot of memory. Use `--summary-base-url` for an Ollama server running on another machine.

### Summarizing an existing run

`echoes summarize RUN_DIR` summarizes a run that is already transcribed, without running Whisper or pyannote again:

```bash
echoes summarize echoes_runs/interview_2026-09-17_15-07-00 --provider gemini
echoes summarize echoes_runs/interview_2026-09-17_15-07-00 --provider ollama --model qwen3
```

- It uses `transcript.txt`, **including your edits**. For instance, replace `Speaker 1` with the person's name before summarizing.
- Existing summaries are kept: new ones are written to `summary_2.md`, `summary_3.md`, and so on.
- Each summary file starts with a small header that records the provider, the exact model version and the date.
- If the summary step of a run fails (quota, network...), the transcript is kept and echoes prints the `echoes summarize` command to retry.

## Output

Each run creates its own folder, by default in an `echoes_runs` folder next to the audio file:

```text
echoes_runs/
└── interview_2026-09-17_15-07-00/
    ├── transcript.txt       readable transcript, one paragraph per speaker turn
    ├── transcript.srt       subtitles, one cue per sentence
    ├── transcript.json      speaker turns, for further processing
    ├── summary.md           summary, with --summary only
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
5. **Summary** (optional): the readable transcript is sent to the chosen LLM with instructions to find the questions, or the topics in an open discussion. Missing keys are detected before the transcription starts, and temporary errors (rate limits, overloaded service) are retried for about two minutes.

## Tips

- **GPU memory**: `large-v3` needs noticeably more memory than `large-v3-turbo`. On a 5 GB card (Quadro P2200), `large-v3` did not fit while `large-v3-turbo` runs fine. Closing GPU-hungry applications (browsers, video calls, GIS software) also helps.
- **Too many speakers?** A short noise or a laugh can create a spurious speaker. Re-run with `--num-speakers` or `--max-speakers`, reusing the transcription to save time.
- **Wrong language?** Detection only listens to the first 30 seconds. Set `--language` if the recording starts with silence or music.
- **Misspelled names?** Add them to `--hotwords`.
- **"HTTP 503" or "HTTP 429" from the summary provider?** Free tiers are often overloaded or rate limited. Wait a little, then run `echoes summarize` on the run folder, or pick another model.

## Privacy

- Transcription and diarization happen locally. Models are downloaded once from Hugging Face and cached.
- pyannote.audio 4 sends anonymous usage metrics (such as file duration and number of speakers) by default. **echoes turns them off**, unless you explicitly set `PYANNOTE_METRICS_ENABLED=1`.
- **Summaries are the exception**: with any provider other than `ollama`, the transcript text (not the audio) is sent to that provider and handled under its terms. Free tiers may use your data to improve their products; for instance, see the [Gemini API terms](https://ai.google.dev/gemini-api/terms) about unpaid services. For confidential interviews, use `ollama`, or a paid plan whose terms suit you.
- Recordings and transcripts often contain personal data. The `.gitignore` excludes audio files and run folders, but handle them according to your local regulations (e.g. GDPR).

## Development

```bash
pip install -e ".[dev]"
pytest
```

The unit tests cover the alignment, export and summary logic. They need neither a GPU, nor any model, nor network access: calls to LLM providers are simulated.

## License

[MIT](LICENSE). The models have their own terms: see the model cards of [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo) and [pyannote speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1), and the terms of the LLM provider you use for summaries.

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [OpenAI Whisper](https://github.com/openai/whisper) and [pyannote.audio](https://github.com/pyannote/pyannote-audio).
