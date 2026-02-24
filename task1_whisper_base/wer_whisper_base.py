"""
Task 1: WER Calculation for Whisper Base transcriptions.

Reference  = Transcript (used as-is, no normalization)
Hypothesis = Normalised_Transcript (normalized via jiwer)

Uses only the 'test' split.
"""

import os

import jiwer
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# --------------- Shared HuggingFace cache ---------------
HF_CACHE = os.path.join(os.path.expanduser("~"), "hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)
os.environ.setdefault("HF_DATASETS_CACHE", HF_CACHE)

# --------------- Normalization (applied ONLY to hypothesis) ---------------
hypothesis_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])

# --------------- Load dataset (test split only) ---------------
print("Loading dataset raianand/TIE_shorts (test split) ...")
print(f"  Cache directory: {HF_CACHE}")
ds = load_dataset("raianand/TIE_shorts", split="test", cache_dir=HF_CACHE)
print(f"  Loaded {len(ds)} samples\n")

all_rows = []

print(f"--- Processing test split ({len(ds)} samples) ---")

for sample in tqdm(ds, desc="test"):
    ref = sample["Transcript"]
    hyp = sample["Normalised_Transcript"]

    if not ref:
        continue

    ref = ref.strip()
    if not ref:
        continue

    hyp_norm = hypothesis_transform(hyp) if hyp else ""

    # Per-sample WER — guard against empty hypothesis
    if not hyp_norm:
        sample_wer = 1.0
    else:
        sample_wer = jiwer.wer(ref, hyp_norm)

    all_rows.append({
        "split": "test",
        "ID": sample.get("ID", ""),
        "reference": ref,
        "hypothesis": hyp_norm,
        "wer": round(sample_wer, 4),
        "Speaker_ID": sample.get("Speaker_ID", ""),
        "Gender": sample.get("Gender", ""),
        "Speech_Class": sample.get("Speech_Class", ""),
        "Native_Region": sample.get("Native_Region", ""),
        "Speech_Duration_seconds": sample.get("Speech_Duration_seconds") or "",
    })

# --------------- Corpus-level WER (including empty hypotheses) ---------------
df = pd.DataFrame(all_rows)
all_refs = df["reference"].tolist()
all_hyps = df["hypothesis"].tolist()

valid_refs = [r for r, h in zip(all_refs, all_hyps) if h]
valid_hyps = [h for h in all_hyps if h]
if valid_refs:
    output = jiwer.process_words(valid_refs, valid_hyps)
    corpus_errors = output.substitutions + output.deletions + output.insertions
else:
    corpus_errors = 0
# Add word counts from empty-hypothesis samples (all deletions)
empty_ref_words = sum(len(r.split()) for r, h in zip(all_refs, all_hyps) if not h)
total_ref_words = sum(len(r.split()) for r in all_refs)
corpus_errors += empty_ref_words
overall_wer = corpus_errors / total_ref_words if total_ref_words else 0.0

print("=" * 60)
print(f"Corpus-level WER (whisper-base, test): {overall_wer:.4f}  ({overall_wer*100:.2f}%)")
print(f"Total samples processed: {len(df)}")
print("=" * 60)

# --------------- Save results ---------------
output_csv = os.path.join(os.path.dirname(__file__), "..", "results", "wer_base.csv")
df.to_csv(output_csv, index=False)
print(f"\nPer-sample results saved to: {output_csv}")
