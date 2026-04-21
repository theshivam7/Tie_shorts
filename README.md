# TIE Shorts -- Whisper WER Evaluation

Word Error Rate (WER) evaluation of OpenAI Whisper **Base**, **Medium**, and **Large** models on Indian English academic lectures, using 4 normalization modes to isolate the impact of text normalization on measured accuracy.

## Key Results

All numbers below use `whisper_normalized` mode (gold standard — symmetric normalization on both reference and hypothesis using `EnglishTextNormalizer`).

| Model | Corpus WER | Mean WER | Median WER | Std Dev |
|-------|:----------:|:--------:|:----------:|:-------:|
| Whisper Base (74M) | 16.92% | 18.37% | 13.21% | 21.11% |
| **Whisper Medium (769M)** | **14.53%** | **15.75%** | **10.91%** | **20.35%** |
| Whisper Large (~1.5B) | 16.02% | 17.69% | 11.30% | 24.65% |

**Key finding:** Whisper Medium outperforms Whisper Large on Indian English speech. Large is more prone to hallucinations on accented English and degrades on longer audio clips (>60s).

## Dataset

[raianand/TIE_shorts](https://huggingface.co/datasets/raianand/TIE_shorts) — 986 samples from the `test` split of the TIE (Talks in Indian English) dataset. NPTEL-style academic lectures with rich metadata:

- **Speakers:** 928 male, 58 female
- **Speech rate:** FAST (413), SLOW (374), AVG (199)
- **Region:** SOUTH (363), EAST (352), NORTH (202), WEST (69)
- **Domains:** Engineering and Non-Engineering disciplines

## Evaluation Modes

We evaluate under 4 normalization modes to understand how normalization choices affect measured WER:

| Mode | Reference Source | Reference Normalization | Hypothesis Normalization | Symmetric? |
|------|-----------------|------------------------|--------------------------|:----------:|
| `raw` | `Transcript` | None | None | Yes |
| `normalized` | `Normalised_Transcript` | None (as-is) | `EnglishTextNormalizer` | No |
| `double_normalized` | `Normalised_Transcript` | `EnglishTextNormalizer` | `EnglishTextNormalizer` | Yes |
| `whisper_normalized` | `Transcript` | `EnglishTextNormalizer` | `EnglishTextNormalizer` | Yes |

**Use `whisper_normalized` for all primary comparisons.** It is the only mode that is both symmetric and free of the dataset's normalization errors.

## Text Normalization — EnglishTextNormalizer

We use OpenAI Whisper's built-in `EnglishTextNormalizer` instead of basic jiwer transforms. Here is what it does:

| Transform | Example |
|-----------|---------|
| Lowercase | `"The Second"` → `"the second"` |
| Expand contractions | `"don't"` → `"do not"`, `"it's"` → `"it is"` |
| Convert numbers to words | `"1st"` → `"first"`, `"100"` → `"one hundred"` |
| Expand symbols | `"$100"` → `"one hundred dollars"`, `"5%"` → `"five percent"` |
| Expand abbreviations | `"Dr."` → `"doctor"`, `"Mr."` → `"mister"` |
| Remove punctuation | `"Hello, world."` → `"hello world"` |
| Normalize whitespace | `"too   many"` → `"too many"` |

### Why the dataset's `Normalised_Transcript` was incorrect

The dataset's `Normalised_Transcript` column contains systematic bad number conversions:

```
Original Transcript:      "the 1st time"
Normalised_Transcript:    "the one s t time"   ← wrong (split character by character)
EnglishTextNormalizer:    "the first time"      ← correct
```

This is why `normalized` and `double_normalized` modes show higher WER than `whisper_normalized` — the reference still contains artifacts like `"one s t"`. The `whisper_normalized` mode bypasses this entirely by starting from the original clean `Transcript`.

## WER Formula

```
WER = (Substitutions + Deletions + Insertions) / Total Reference Words
```

- **Corpus WER** — micro-averaged: total errors across all samples / total reference words
- **Mean/Median WER** — per-sample statistics (can differ from corpus WER)
- WER > 100% is possible when insertions exceed reference length (hallucinations)

## Analysis Breakdown

### By Speech Rate (whisper_normalized)

| Speech Class | n | Base | Medium | Large |
|:------------:|:-:|:----:|:------:|:-----:|
| FAST | 413 | 15.76% | 13.32% | 13.85% |
| AVG | 199 | 15.46% | 13.32% | 15.48% |
| SLOW | 374 | 19.40% | 16.91% | 19.31% |

Slow speech consistently has the highest WER — Whisper halluccinates during pauses.

### By Region (whisper_normalized)

| Region | n | Base | Medium | Large |
|:------:|:-:|:----:|:------:|:-----:|
| NORTH | 202 | 16.87% | 14.89% | 15.89% |
| SOUTH | 363 | 17.50% | 15.11% | 15.78% |
| EAST | 352 | 16.38% | 13.58% | 16.78% |
| WEST | 69 | 16.56% | 14.77% | 14.25% |

### By Gender (whisper_normalized)

| Gender | n | Base | Medium | Large |
|:------:|:-:|:----:|:------:|:-----:|
| Female | 58 | 13.88% | 12.10% | 14.87% |
| Male | 928 | 17.11% | 14.67% | 16.09% |

Female speakers have consistently lower WER across all models, though the dataset is heavily male-dominated (928M vs 58F).

## Project Structure

```
.
├── utils/
│   ├── normalize.py             # 4-mode normalization using EnglishTextNormalizer
│   ├── transcribe.py            # Audio extraction + Whisper inference pipeline
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
│   └── compare_all.py           # Cross-model comparison, charts, markdown report
├── results/
│   ├── raw/                     # Per-sample WER CSVs — no normalization
│   ├── normalized/              # Per-sample WER CSVs — dataset normalization
│   ├── double_normalized/       # Per-sample WER CSVs — re-normalized
│   ├── whisper_normalized/      # Per-sample WER CSVs — gold standard
│   ├── wer_summary_{model}.csv  # Per-model stats across all 4 modes
│   ├── wer_summary_{model}.md   # Human-readable version of above
│   ├── top_20_high_wer_*.csv    # Top 20 highest-WER samples per model per mode
│   └── analysis/                # Cross-model charts, breakdowns, summary report
├── logs/                        # Job execution logs (NSCC)
├── summary.md                   # Full analysis report
└── README.md
```

## CSV Column Schema

Each per-sample CSV contains:

| Column | Description |
|--------|-------------|
| `mode` | Evaluation mode name |
| `reference_source` | Dataset column used as reference (`Transcript` or `Normalised_Transcript`) |
| `reference_raw` | Reference text **before** normalization |
| `reference` | Reference text **after** normalization (used for WER) |
| `hypothesis_raw` | Raw Whisper output before normalization |
| `hypothesis` | Whisper output after normalization (used for WER) |
| `wer` | Per-sample WER |
| + metadata | `Speaker_ID`, `Gender`, `Speech_Class`, `Native_Region`, `Speech_Duration_seconds`, `Discipline_Group`, `Topic` |

## How to Run

Requires a single conda environment with all dependencies:

```bash
conda create -n whisper python=3.10 -y
conda activate whisper
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install openai-whisper jiwer pandas librosa datasets tqdm matplotlib numpy
conda install -c conda-forge ffmpeg -y
```

Run each model sequentially:

```bash
# Step 1: Whisper Base (~12 min on A100)
python task1_whisper_base/wer_whisper_base.py

# Step 2: Whisper Medium (~35 min on A100)
python task2_whisper_medium/wer_whisper_medium.py

# Step 3: Whisper Large (~90 min on A100)
python task3_whisper_large/wer_whisper_large.py

# Step 4: Cross-model analysis and charts
python analysis/compare_all.py
```

All scripts support **crash recovery** — checkpoints are saved every 200 samples and automatically resumed on restart.

## Analysis Outputs

| File | Description |
|------|-------------|
| `results/wer_summary_{model}.csv` | Corpus WER, mean, median, std, P90, P95 across all 4 modes |
| `results/wer_summary_{model}.md` | Same as above in readable markdown table |
| `results/top_20_high_wer_{model}_{mode}.csv` | 20 hardest samples per model per mode |
| `results/analysis/wer_summary.csv` | 3×4 corpus WER matrix (models × modes) |
| `results/analysis/comparison_by_region.csv` | WER breakdown by Native_Region |
| `results/analysis/comparison_by_speech_class.csv` | WER breakdown by Speech_Class |
| `results/analysis/comparison_by_gender.csv` | WER breakdown by Gender |
| `results/analysis/comparison_by_discipline.csv` | WER breakdown by Discipline_Group |
| `results/analysis/comparison_by_duration.csv` | WER breakdown by audio duration bucket |
| `results/analysis/wer_by_model_and_mode.png` | Grouped bar chart across models and modes |
| `results/analysis/wer_distribution.png` | WER histogram per model |
| `results/analysis/wer_by_region.png` | WER by region per model |
| `results/analysis/wer_by_speech_class.png` | WER by speech class per model |
| `results/analysis/wer_by_duration.png` | WER by audio duration per model |
| `results/analysis/summary_report.md` | Auto-generated full analysis report |

## Tech Stack

- Python 3.10+
- [openai-whisper](https://github.com/openai/whisper) — ASR models
- [jiwer](https://github.com/jitsi/jiwer) — WER computation
- [HuggingFace Datasets](https://huggingface.co/docs/datasets) — data loading
- pandas, matplotlib, librosa, torch
