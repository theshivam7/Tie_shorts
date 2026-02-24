# TIE Shorts — Whisper WER Evaluation

Word Error Rate (WER) evaluation of OpenAI Whisper **Base** and **Medium** models on Indian English academic lectures.

## Key Results

| Model | Mean WER | Median WER | Perfect Scores |
|-------|:--------:|:----------:|:--------------:|
| **Whisper Base** | **0.1882 (18.82%)** | **0.1719 (17.19%)** | **40** |
| Whisper Medium | 0.2940 (29.40%) | 0.2600 (26.00%) | 3 |

Whisper Base outperforms Medium on **81.3%** of 986 test samples. See [`summary.md`](summary.md) for the full analysis.

## Dataset

[raianand/TIE_shorts](https://huggingface.co/datasets/raianand/TIE_shorts) — 986 samples from the `test` split. Indian English NPTEL-style academic lectures with metadata for speaker gender, speech speed, native region, and discipline.

## Project Structure

```
.
├── task1_whisper_base/
│   ├── wer_whisper_base.py    # WER computation (base model)
│   ├── requirements.txt
│   └── setup.sh
├── task2_whisper_medium/
│   ├── wer_whisper_medium.py  # Transcription + WER (medium model)
│   ├── requirements.txt
│   └── setup.sh
├── results/
│   ├── wer_base.csv           # Per-sample WER — base (986 rows)
│   ├── wer_medium.csv         # Per-sample WER — medium (986 rows)
│   ├── wer_comparison.csv     # Side-by-side comparison with diff
│   └── top_20_high_wer.csv    # Top 20 highest-WER sentences (both models)
├── summary.md                 # Detailed analysis report
└── README.md
```

## Quick Start

```bash
# Task 1: Whisper Base WER
cd task1_whisper_base
bash setup.sh
conda activate tie_wer_base
python wer_whisper_base.py

# Task 2: Whisper Medium WER (requires GPU)
cd task2_whisper_medium
bash setup.sh
conda activate tie_wer_medium
python wer_whisper_medium.py
```

## Methodology

- **Reference:** `Transcript` field from dataset (no normalization)
- **Hypothesis (Task 1):** `Normalised_Transcript` from dataset (Whisper Base output)
- **Hypothesis (Task 2):** Live inference using `openai/whisper-medium` on raw audio
- **Normalization (hypothesis only):** lowercase, remove punctuation, collapse spaces, strip
- **WER Library:** [jiwer](https://github.com/jitsi/jiwer)
