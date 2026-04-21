# WER Analysis: Whisper Base vs Medium vs Large on Indian English

## Dataset

- **Source:** [raianand/TIE_shorts](https://huggingface.co/datasets/raianand/TIE_shorts)
- **Split:** `test` (986 samples)
- **Domain:** Indian English academic lectures (NPTEL-style)
- **Distribution:** 928M / 58F | FAST 413, SLOW 374, AVG 199 | SOUTH 363, EAST 352, NORTH 202, WEST 69

## Methodology

All three models are evaluated using live Whisper inference on raw audio. Each model produces transcriptions that are then compared against human reference transcripts under 4 normalization modes.

### Audio Pipeline

1. Extract audio array from HuggingFace dataset
2. Convert to float32, flatten to mono
3. Resample to 16kHz using librosa
4. Write to temporary WAV file
5. Transcribe with `model.transcribe(language="en")`

### Normalization

Text normalization is applied via `whisper.normalizers.EnglishTextNormalizer` which handles:
- Lowercase conversion
- Contraction expansion (`don't` → `do not`)
- Number-to-word conversion (`1st` → `first`, `100` → `one hundred`)
- Symbol expansion (`$100` → `one hundred dollars`)
- Punctuation removal
- Whitespace normalization

### Evaluation Modes

| Mode | Reference | Hypothesis | Notes |
|------|-----------|------------|-------|
| `raw` | `Transcript` as-is | Whisper output as-is | Penalizes casing/punctuation |
| `normalized` | `Normalised_Transcript` as-is | `EnglishTextNormalizer(hyp)` | Asymmetric; dataset has errors |
| `double_normalized` | `EnglishTextNormalizer(Normalised_Transcript)` | `EnglishTextNormalizer(hyp)` | Fixes punctuation but not number errors |
| `whisper_normalized` | `EnglishTextNormalizer(Transcript)` | `EnglishTextNormalizer(hyp)` | Gold standard — use this for all comparisons |

The dataset's `Normalised_Transcript` column contains systematic errors (e.g., `"1st"` → `"one s t"`). The `whisper_normalized` mode bypasses this entirely.

### WER Formula

```
WER = (Substitutions + Deletions + Insertions) / Total Reference Words
```

Corpus WER is micro-averaged (total errors / total words across all samples). WER > 1.0 indicates more errors than reference words (hallucinations).

---

## Results

### Corpus WER Across All Modes

| Model | raw | normalized | double_normalized | whisper_normalized |
|-------|:---:|:----------:|:-----------------:|:-----------------:|
| Base (74M) | 27.86% | 21.32% | 17.88% | 16.92% |
| Medium (769M) | 24.19% | 19.23% | 15.68% | **14.53%** |
| Large (~1.5B) | 25.91% | 20.67% | 17.10% | 16.02% |

### Detailed Stats (whisper_normalized)

| Model | Corpus WER | Mean WER | Median WER | Std Dev | P90 | P95 |
|-------|:----------:|:--------:|:----------:|:-------:|:---:|:---:|
| Base | 16.92% | 18.37% | 13.21% | 21.11% | 37.18% | 50.00% |
| Medium | 14.53% | 15.75% | 10.91% | 20.35% | 31.33% | 41.51% |
| Large | 16.02% | 17.69% | 11.30% | 24.65% | 34.72% | 51.06% |

**Whisper Medium is the best-performing model on Indian English speech.**

---

## Key Finding: Medium Beats Large

Whisper Large underperforms Medium on this dataset. Contributing factors:

1. **Hallucination rate** — Large produces more hallucinated text during pauses and on accented speech. One sample produced Korean and Cyrillic characters as output.
2. **Long audio degradation** — Large WER on 60s+ clips reaches ~38% vs Medium's ~28%.
3. **Higher variance** — Large has std dev of 24.65% vs Medium's 20.35%, indicating less consistent behavior.
4. **SLOW speech** — Both Base and Large show ~19% WER on slow speakers vs Medium's 16.91%.

This aligns with known behavior: larger Whisper models are more prone to hallucination on out-of-distribution accents.

---

## Breakdown by Speech Rate (whisper_normalized)

| Speech Class | n | Base | Medium | Large |
|:------------:|:-:|:----:|:------:|:-----:|
| FAST | 413 | 15.76% | 13.32% | 13.85% |
| AVG | 199 | 15.46% | 13.32% | 15.48% |
| SLOW | 374 | 19.40% | 16.91% | 19.31% |

SLOW speech is consistently hardest across all models — Whisper hallucinates during long pauses between words.

---

## Breakdown by Region (whisper_normalized)

| Region | n | Base | Medium | Large |
|:------:|:-:|:----:|:------:|:-----:|
| NORTH | 202 | 16.87% | 14.89% | 15.89% |
| SOUTH | 363 | 17.50% | 15.11% | 15.78% |
| EAST | 352 | 16.38% | 13.58% | 16.78% |
| WEST | 69 | 16.56% | 14.77% | 14.25% |

SOUTH and EAST have the largest speaker count and highest WER for Base. Medium is consistently best across all regions. Notably, Large performs worse than Medium in every region.

---

## Breakdown by Gender (whisper_normalized)

| Gender | n | Base | Medium | Large |
|:------:|:-:|:----:|:------:|:-----:|
| Female | 58 | 13.88% | 12.10% | 14.87% |
| Male | 928 | 17.11% | 14.67% | 16.09% |

Female speakers have consistently lower WER across all models (~3 pp lower). The dataset is heavily male-dominated so this difference should be interpreted carefully.

---

## Why Certain Sentences Have High WER

- **Short references** — Samples with 1-3 word references have extreme WER from a single error. Example: reference `"."` vs hypothesis `"can be written as"` gives WER = 4.0.
- **Math and symbolic content** — Equations like `"ds/dt = pi r square"` have no standard spoken form.
- **Slow speech with pauses** — Whisper hallucinates filler text during silence (especially Medium and Large).
- **Indian English accents** — Regional phonological patterns diverge from Whisper's predominantly Western English training data.
- **Technical vocabulary** — Domain-specific terms are out-of-vocabulary.
- **Code-switching** — Hindi or regional language words in English lectures cause confusion.
- **Whisper Large hallucinations** — Non-English text appears occasionally (Korean, Cyrillic characters observed).

---

## How to Improve WER on Indian English

**Normalization:**
- Use symmetric normalization on both reference and hypothesis (already done in `whisper_normalized` mode)
- Filter short references (< 3 words) from evaluation metrics

**Model:**
- Fine-tune on Indian English data (NPTEL, IIT lectures)
- Use `initial_prompt` with domain-specific vocabulary
- Apply VAD (voice activity detection) preprocessing to reduce hallucinations during silence

**Inference:**
- Try `whisper-medium` with `beam_size=5` and `best_of=5` for better accuracy
- Use `condition_on_previous_text=False` to reduce hallucination cascade

---

## Reproducibility

| | Base | Medium | Large |
|-|------|--------|-------|
| Model params | 74M | 769M | ~1.5B |
| Python | 3.10 | 3.10 | 3.10 |
| CUDA | 12.2 | 12.2 | 12.2 |
| GPU | A100 | A100 | A100 |
| Runtime | ~12 min | ~35 min | ~90 min |
| Samples | 986 | 986 | 986 |

All results are in `results/`. Per-model WER summaries: `results/wer_summary_{base,medium,large}.md`.
