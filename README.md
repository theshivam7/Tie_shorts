# TIE Shorts -- ASR WER Evaluation

Word Error Rate (WER) evaluation of **4 ASR systems** on Indian English academic lectures from the TIE (Talks in Indian English) dataset. Evaluated across 4 normalization modes to compare reference sources and the effect of normalization on measured WER.

**ASR Systems evaluated:**
- OpenAI Whisper Base (74M parameters)
- OpenAI Whisper Medium (769M parameters)
- OpenAI Whisper Large (~1.5B parameters)
- YouTube Auto-generated Captions (Google ASR)

## Dataset

[raianand/TIE_shorts](https://huggingface.co/datasets/raianand/TIE_shorts) — 986 samples from the `test` split. Indian English NPTEL-style academic lectures. Video IDs in this dataset are valid YouTube IDs.

| Attribute | Values |
|-----------|--------|
| Speakers | 928 male, 58 female |
| Speech rate | FAST (413), SLOW (374), AVG (199) |
| Region | SOUTH (363), EAST (352), NORTH (202), WEST (69) |
| Disciplines | Engineering (691), Non-Engineering (295) |

## Evaluation Modes

4 modes covering 2 reference sources × 2 normalization states — all symmetric:

| Mode | Reference Source | Before Norm | After Norm | Purpose |
|------|-----------------|-------------|------------|---------|
| `transcript_raw` | `Transcript` | as-is | as-is | Upper bound baseline |
| `transcript_clean` | `Transcript` | `Transcript` | normalized | **Gold standard — paper primary** |
| `hf_raw` | `Normalised_Transcript` | as-is | as-is | HuggingFace normalization quality check |
| `hf_clean` | `Normalised_Transcript` | `Normalised_Transcript` | normalized | HF + our normalizer |

**Why 4 modes?** The dataset's `Normalised_Transcript` contains errors (e.g. `"1st"` → `"one s t"`). By comparing all 4 modes we can quantify the impact and justify why `transcript_clean` gives the fairest WER.

## Normalization Pipeline (`*_clean` modes)

Applied **symmetrically** to both reference and hypothesis (forward normalization — standard for WER):

| Step | Example |
|------|---------|
| Unicode NFC | fix encoding artifacts |
| Expand contractions | `"don't"` → `"do not"` |
| Fix possessives | `"Bernoulli's"` → `"bernoulli s"` |
| Ordinals → words | `"1st"` → `"first"`, `"2nd"` → `"second"` |
| Cardinals → words | `"100"` → `"one hundred"`, `"60,000"` → `"sixty thousand"` |
| Lowercase | `"The First"` → `"the first"` |
| Remove punctuation | `"hello,"` → `"hello"` |
| Normalize whitespace | collapse multiple spaces |

## WER Formula

```
WER = (Substitutions + Deletions + Insertions) / Total Reference Words
```

Corpus WER = micro-averaged (total errors / total words across all samples). WER > 1.0 is possible when hallucinations exceed reference length.

## Pipeline Architecture

### Stage 1 — ASR Transcription (run once per model)

Each script saves raw outputs to `results/stage1_raw_transcripts/`. Never re-run unless changing the model.

```
Stage 1 output columns: transcript_raw, normalised_transcript_raw, hypothesis_raw, [caption_type for YouTube]
```

| Script | Model | GPU needed | Time (A100) |
|--------|-------|-----------|-------------|
| `task1_whisper_base/wer_whisper_base.py` | Whisper Base | Yes | ~12 min |
| `task2_whisper_medium/wer_whisper_medium.py` | Whisper Medium | Yes | ~35 min |
| `task3_whisper_large/wer_whisper_large.py` | Whisper Large | Yes | ~90 min |
| `task4_youtube_captions/fetch_youtube_captions.py` | YouTube captions | No | ~30 min |

### Stage 2 — Normalization + WER (re-run freely, no GPU)

```bash
python normalize_and_score.py    # ~1 min, no GPU
python analysis/compare_all.py   # charts and breakdowns
```

## Quick Start

```bash
# Install dependencies
conda activate whisper
pip install openai-whisper torch librosa jiwer pandas tqdm numpy num2words datasets
pip install youtube-transcript-api  # for YouTube captions

# Run full pipeline automatically (NSCC PBS job)
qsub run_pipeline.pbs

# Or run manually step by step
python task1_whisper_base/wer_whisper_base.py
python task2_whisper_medium/wer_whisper_medium.py
python task3_whisper_large/wer_whisper_large.py
python task4_youtube_captions/fetch_youtube_captions.py
python normalize_and_score.py
python analysis/compare_all.py
```

## Results Folder Structure

```
results/
  stage1_raw_transcripts/           ← raw ASR outputs (read-only after run)
    wer_base_raw.csv
    wer_medium_raw.csv
    wer_large_raw.csv
    wer_youtube_raw.csv             ← includes caption_type column
  stage2_processed/                 ← WER results per mode
    transcript_raw/
      wer_base_transcript_raw.csv
      wer_medium_transcript_raw.csv
      wer_large_transcript_raw.csv
      wer_youtube_transcript_raw.csv
    transcript_clean/               ← gold standard results
      wer_base_transcript_clean.csv
      ...
    hf_raw/
      ...
    hf_clean/
      ...
    wer_summary_all_models.csv      ← 4 models × 4 modes matrix
    wer_summary_all_models.md
    top_20_high_wer_{model}_{mode}.csv
  analysis/
    wer_summary.csv
    summary_report.md
    comparison_by_region.csv
    comparison_by_speech_class.csv
    comparison_by_gender.csv
    comparison_by_discipline.csv
    comparison_by_duration.csv
    wer_by_model_and_mode.png
    wer_distribution.png
    wer_by_region.png
    wer_by_speech_class.png
    wer_by_duration.png
```

## CSV Column Schema

Each Stage 2 result CSV contains:

| Column | Description |
|--------|-------------|
| `model` | ASR model name (`base`, `medium`, `large`, `youtube`) |
| `mode` | Evaluation mode name |
| `reference_source` | Dataset column used as reference |
| `reference_raw` | Reference text **before** normalization |
| `reference` | Reference text **after** normalization (used for WER) |
| `hypothesis_raw` | ASR output **before** normalization |
| `hypothesis` | ASR output **after** normalization (used for WER) |
| `wer` | Per-sample WER |
| + metadata | `Speaker_ID`, `Gender`, `Speech_Class`, `Native_Region`, `Speech_Duration_seconds`, `Discipline_Group`, `Topic` |

In `*_raw` modes: `reference_raw == reference` and `hypothesis_raw == hypothesis`.

## Project Structure

```
.
├── utils/
│   ├── normalize.py              # 4-mode normalization logic
│   ├── transcribe.py             # Audio processing + Whisper inference
│   ├── wer_compute.py            # Per-sample and corpus WER
│   └── io_helpers.py             # Dataset loading, I/O, checkpointing
├── task1_whisper_base/           # Stage 1: Whisper Base
├── task2_whisper_medium/         # Stage 1: Whisper Medium
├── task3_whisper_large/          # Stage 1: Whisper Large
├── task4_youtube_captions/       # Stage 1: YouTube captions (no GPU)
├── normalize_and_score.py        # Stage 2: Normalization + WER
├── analysis/
│   └── compare_all.py            # Cross-model comparison + charts
├── run_pipeline.pbs              # NSCC PBS job (full automated pipeline)
└── results/
    ├── stage1_raw_transcripts/
    ├── stage2_processed/
    └── analysis/
```

## Tech Stack

- Python 3.10+
- [openai-whisper](https://github.com/openai/whisper) — ASR models
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) — YouTube captions
- [jiwer](https://github.com/jitsi/jiwer) — WER computation
- [num2words](https://github.com/savoirfairelinux/num2words) — number normalization
- HuggingFace Datasets, pandas, matplotlib, librosa, torch
