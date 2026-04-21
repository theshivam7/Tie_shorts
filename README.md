# TIE Shorts -- Whisper WER Evaluation

Word Error Rate (WER) evaluation of OpenAI Whisper **Base**, **Medium**, and **Large** models on Indian English academic lectures, using 4 normalization modes to isolate the impact of text normalization on WER.

## Dataset

[raianand/TIE_shorts](https://huggingface.co/datasets/raianand/TIE_shorts) -- 986 samples from the `test` split. Indian English NPTEL-style academic lectures with metadata for speaker gender, speech speed, native region, discipline, and duration.

## Evaluation Modes

| Mode | Reference | Hypothesis | Purpose |
|------|-----------|------------|---------|
| `raw` | Transcript (as-is) | Whisper output (as-is) | Baseline with no normalization |
| `normalized` | Normalised_Transcript (as-is) | EnglishTextNormalizer(hyp) | Shows impact of dataset's normalization |
| `double_normalized` | EnglishTextNormalizer(Normalised_Transcript) | EnglishTextNormalizer(hyp) | Fixes dataset normalization errors |
| `whisper_normalized` | EnglishTextNormalizer(Transcript) | EnglishTextNormalizer(hyp) | Gold standard: symmetric normalization |

The `whisper_normalized` mode uses OpenAI's `EnglishTextNormalizer` on both sides, bypassing the dataset's buggy `Normalised_Transcript` column (e.g., "second" was incorrectly normalized to "two n d").

## Project Structure

```
.
├── utils/
│   ├── normalize.py             # 4-mode normalization using EnglishTextNormalizer
│   ├── transcribe.py            # Audio extraction + Whisper inference
│   ├── wer_compute.py           # Per-sample and corpus WER computation
│   └── io_helpers.py            # Dataset loading, CSV I/O, checkpointing
├── task1_whisper_base/
│   ├── wer_whisper_base.py      # Whisper base evaluation (all 4 modes)
│   ├── requirements.txt
│   └── setup.sh
├── task2_whisper_medium/
│   ├── wer_whisper_medium.py    # Whisper medium evaluation (all 4 modes)
│   ├── requirements.txt
│   └── setup.sh
├── task3_whisper_large/
│   ├── wer_whisper_large.py     # Whisper large evaluation (all 4 modes)
│   ├── requirements.txt
│   └── setup.sh
├── analysis/
│   └── compare_all.py           # Cross-model comparison, charts, report
├── results/
│   ├── raw/                     # WER CSVs using raw transcripts
│   ├── normalized/              # WER CSVs using dataset normalization
│   ├── double_normalized/       # WER CSVs using re-normalized transcripts
│   ├── whisper_normalized/      # WER CSVs using gold-standard normalization
│   └── analysis/                # Summary tables, charts, markdown report
├── summary.md
└── README.md
```

## Quick Start

Each model is run sequentially. Run one, validate results, then proceed to the next.

```bash
# Step 1: Whisper Base
cd task1_whisper_base
bash setup.sh
conda activate tie_wer_base
python wer_whisper_base.py

# Step 2: Whisper Medium
cd ../task2_whisper_medium
bash setup.sh
conda activate tie_wer_medium
python wer_whisper_medium.py

# Step 3: Whisper Large
cd ../task3_whisper_large
bash setup.sh
conda activate tie_wer_large
python wer_whisper_large.py

# Step 4: Cross-model analysis
cd ../analysis
python compare_all.py
```

## Methodology

- **WER Formula:** `WER = (Substitutions + Deletions + Insertions) / Total_Reference_Words`
- **Corpus WER:** Micro-averaged across all samples (total errors / total reference words)
- **Per-sample stats:** Mean, median, std, P90, P95 WER reported for each mode
- **Normalizer:** `whisper.normalizers.EnglishTextNormalizer` (handles contractions, number words, casing, punctuation)
- **WER Library:** [jiwer](https://github.com/jitsi/jiwer)
- **Audio processing:** Resample to 16kHz mono, transcribe via `model.transcribe()`
- **Crash recovery:** Checkpoints saved every 200 samples with automatic resume

## Analysis Outputs

After running all three models and the analysis script:

| Output | Description |
|--------|-------------|
| `wer_summary.csv` | 3x4 corpus WER matrix (models x modes) |
| `comparison_by_region.csv` | WER breakdown by Native_Region |
| `comparison_by_speech_class.csv` | WER breakdown by Speech_Class |
| `comparison_by_gender.csv` | WER breakdown by Gender |
| `comparison_by_discipline.csv` | WER breakdown by Discipline_Group |
| `comparison_by_duration.csv` | WER breakdown by duration buckets |
| `top_20_high_wer_all_models.csv` | Top 20 worst sentences per model |
| `wer_by_model_and_mode.png` | Grouped bar chart: WER across models and modes |
| `wer_distribution.png` | WER distribution histogram per model |
| `wer_by_region.png` | WER by region per model |
| `wer_by_speech_class.png` | WER by speech class per model |
| `wer_by_duration.png` | WER by duration bucket per model |
| `summary_report.md` | Auto-generated markdown report |

## Tech Stack

- Python 3.10+
- OpenAI Whisper (`openai-whisper`)
- jiwer (WER computation)
- HuggingFace Datasets
- pandas, matplotlib, librosa, torch
