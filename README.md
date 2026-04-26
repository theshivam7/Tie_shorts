# TIE Shorts -- Whisper WER Evaluation

Word Error Rate (WER) evaluation of OpenAI Whisper **Base**, **Medium**, and **Large** models on Indian English academic lectures. Evaluated across 4 modes to compare reference sources and the effect of normalization on measured WER.

## Dataset

[raianand/TIE_shorts](https://huggingface.co/datasets/raianand/TIE_shorts) — 986 samples from the `test` split. Indian English NPTEL-style academic lectures with metadata for speaker gender, speech rate, native region, and discipline.

- **Speakers:** 928 male, 58 female
- **Speech rate:** FAST (413), SLOW (374), AVG (199)
- **Region:** SOUTH (363), EAST (352), NORTH (202), WEST (69)

## Evaluation Modes

4 modes covering 2 reference sources × 2 normalization states:

| Mode | Reference Source | Before Normalization | After Normalization | Purpose |
|------|-----------------|---------------------|---------------------|---------|
| `transcript_raw` | `Transcript` | as-is | as-is | Upper bound baseline |
| `transcript_clean` | `Transcript` | `Transcript` | normalized | **Gold standard — paper primary** |
| `hf_raw` | `Normalised_Transcript` | as-is | as-is | HuggingFace normalization as-is |
| `hf_clean` | `Normalised_Transcript` | `Normalised_Transcript` | normalized | HF normalization + our fix |

All modes are **symmetric** — same transformation applied to both reference and hypothesis.

### Why 4 modes?

The dataset's `Normalised_Transcript` column contains errors (e.g. `"1st"` → `"one s t"`). By comparing all 4 modes we can:
- Quantify how much the dataset's normalization errors inflate WER (`hf_raw` vs `transcript_raw`)
- Show that our normalizer fixes those errors (`hf_clean` vs `hf_raw`)
- Report the lowest valid WER using correct reference and correct normalization (`transcript_clean`)

### Normalization Pipeline (`*_clean` modes)

Applied identically to both reference and hypothesis (forward normalization):

| Step | Example |
|------|---------|
| Unicode NFC | fix encoding artifacts |
| Expand contractions | `"don't"` → `"do not"` |
| Fix possessives | `"Bernoulli's"` → `"bernoulli s"` |
| Ordinals → words | `"1st"` → `"first"` |
| Cardinals → words | `"100"` → `"one hundred"`, `"60,000"` → `"sixty thousand"` |
| Lowercase | `"The"` → `"the"` |
| Remove punctuation | `"hello,"` → `"hello"` |
| Normalize whitespace | collapse multiple spaces |

## WER Formula

```
WER = (Substitutions + Deletions + Insertions) / Total Reference Words
```

Corpus WER = micro-averaged across all samples. WER > 1.0 is possible when insertions exceed reference length (hallucinations).

## Two-Stage Pipeline

### Stage 1 — ASR Transcription (GPU required, run once)

Runs Whisper inference and saves raw outputs. Never re-run unless model changes.

```bash
python task1_whisper_base/wer_whisper_base.py     # ~12 min on A100
python task2_whisper_medium/wer_whisper_medium.py  # ~35 min on A100
python task3_whisper_large/wer_whisper_large.py    # ~90 min on A100
```

Output: `results/stage1_raw_transcripts/wer_{base,medium,large}_raw.csv`

Each Stage 1 CSV has: `transcript_raw`, `normalised_transcript_raw`, `hypothesis_raw` — all untouched.

### Stage 2 — Normalization + WER (CPU only, re-run freely)

```bash
python normalize_and_score.py   # ~30 seconds, no GPU
python analysis/compare_all.py  # charts and breakdowns
```

Output: `results/stage2_processed/{mode}/wer_{model}_{mode}.csv`

## Output CSV Schema

Each result CSV contains:

| Column | Description |
|--------|-------------|
| `reference_source` | Which column was used as reference |
| `reference_raw` | Reference text **before** normalization |
| `reference` | Reference text **after** normalization (used for WER) |
| `hypothesis_raw` | Raw Whisper output **before** normalization |
| `hypothesis` | Whisper output **after** normalization (used for WER) |
| `wer` | Per-sample WER |
| + metadata | `Speaker_ID`, `Gender`, `Speech_Class`, `Native_Region`, `Speech_Duration_seconds`, `Discipline_Group`, `Topic` |

In `*_raw` modes: `reference_raw == reference` and `hypothesis_raw == hypothesis`.

## Automated NSCC Run

Submit the full pipeline as a single job:

```bash
qsub run_pipeline.pbs
```

Runs Stage 1 (all 3 models) → Stage 2 → Analysis automatically. Check `logs/pipeline.log`.

## Project Structure

```
.
├── utils/
│   ├── normalize.py             # 4-mode normalization logic
│   ├── transcribe.py            # Audio processing + Whisper inference
│   ├── wer_compute.py           # WER computation utilities
│   └── io_helpers.py            # Dataset loading, I/O, checkpointing
├── task1_whisper_base/          # Stage 1: Whisper Base transcription
├── task2_whisper_medium/        # Stage 1: Whisper Medium transcription
├── task3_whisper_large/         # Stage 1: Whisper Large transcription
├── normalize_and_score.py       # Stage 2: Normalization + WER
├── analysis/
│   └── compare_all.py           # Cross-model comparison + charts
├── run_pipeline.pbs             # NSCC PBS job script (full pipeline)
└── results/
    ├── stage1_raw_transcripts/  # Raw ASR outputs (read-only after run)
    ├── stage2_processed/        # WER results per mode
    └── analysis/                # Charts, breakdowns, summary report
```

## Tech Stack

- Python 3.10+
- [openai-whisper](https://github.com/openai/whisper)
- [jiwer](https://github.com/jitsi/jiwer)
- [num2words](https://github.com/savoirfairelinux/num2words)
- HuggingFace Datasets, pandas, matplotlib, librosa, torch
