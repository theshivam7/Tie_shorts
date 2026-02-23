"""
Task 2: Transcribe with Whisper Medium + WER Calculation + Comparison.

- Transcribes all audio using openai/whisper-medium (loaded locally).
- Reference  = Transcript (as-is)
- Hypothesis = whisper-medium generated transcription (normalized via jiwer)
- Compares WER: whisper-base vs whisper-medium.
- Extracts top 20 highest-WER sentences with analysis guidance.

Uses only the 'test' split.
"""

import argparse
import os
import tempfile
import warnings

import jiwer
import numpy as np
import pandas as pd
from scipy.io import wavfile as scipy_wav
import torch
import whisper
from datasets import load_dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --------------- Shared HuggingFace cache ---------------
HF_CACHE = os.path.join(os.path.expanduser("~"), "hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)
os.environ.setdefault("HF_DATASETS_CACHE", HF_CACHE)

# --------------- CLI arguments ---------------
_parser = argparse.ArgumentParser(description="Task 2: Whisper Medium WER")
_parser.add_argument(
    "--base-csv",
    default=os.path.join(os.path.dirname(__file__), "..", "task1_whisper_base", "wer_whisper_base_results.csv"),
    help="Path to Task 1 whisper-base results CSV",
)
_args = _parser.parse_args()

# --------------- Config ---------------
WHISPER_MODEL = "medium"
BASE_RESULTS_CSV = _args.base_csv

# --------------- Normalization (applied to hypothesis only) ---------------
hypothesis_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])

# --------------- Load Whisper Medium locally ---------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading whisper-{WHISPER_MODEL} on device: {device} ...")
model = whisper.load_model(WHISPER_MODEL, device=device)
print("Model loaded.\n")

# Whisper defaults to fp16 which crashes on CPU
_transcribe_kw = {"language": "en"}
if device == "cpu":
    _transcribe_kw["fp16"] = False

# --------------- Load dataset (test split only) ---------------
print("Loading dataset raianand/TIE_shorts (test split) ...")
print(f"  Cache directory: {HF_CACHE}")
ds = load_dataset("raianand/TIE_shorts", split="test", cache_dir=HF_CACHE)
print(f"  Loaded {len(ds)} samples\n")

all_rows = []

print(f"--- Processing test split ({len(ds)} samples) ---")

for sample in tqdm(ds, desc="test (transcribing)"):
    ref = sample["Transcript"]
    if not ref or not ref.strip():
        continue
    ref = ref.strip()

    # Extract audio and write to a temp wav file for whisper
    audio_data = sample["audio"]
    audio_array = np.array(audio_data["array"], dtype=np.float32)
    sr = audio_data["sampling_rate"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        scipy_wav.write(tmp_path, sr, (audio_array * 32767).astype(np.int16))

    try:
        result = model.transcribe(tmp_path, **_transcribe_kw)
        hyp_raw = result["text"].strip()
    except Exception as e:
        print(f"  [WARN] Failed to transcribe {sample.get('ID', '?')}: {e}")
        hyp_raw = ""
    finally:
        os.unlink(tmp_path)

    hyp_norm = hypothesis_transform(hyp_raw) if hyp_raw else ""

    if not hyp_norm:
        sample_wer = 1.0
    else:
        sample_wer = jiwer.wer(ref, hyp_norm)

    all_rows.append({
        "split": "test",
        "ID": sample.get("ID", ""),
        "reference": ref,
        "hypothesis_medium": hyp_norm,
        "hypothesis_medium_raw": hyp_raw,
        "wer_medium": round(sample_wer, 4),
        "Speaker_ID": sample.get("Speaker_ID", ""),
        "Gender": sample.get("Gender", ""),
        "Speech_Class": sample.get("Speech_Class", ""),
        "Native_Region": sample.get("Native_Region", ""),
        "Speech_Duration_seconds": sample.get("Speech_Duration_seconds") or "",
        "Discipline_Group": sample.get("Discipline_Group", ""),
        "Topic": sample.get("Topic", ""),
    })

    # Periodic checkpoint every 200 samples (crash recovery)
    if len(all_rows) % 200 == 0:
        pd.DataFrame(all_rows).to_csv("wer_whisper_medium_results_partial.csv", index=False)
        print(f"  [checkpoint] {len(all_rows)} samples saved")

# --------------- Compute corpus-level WER (including empty hypotheses) ---------------
df = pd.DataFrame(all_rows)
all_refs = df["reference"].tolist()
all_hyps = df["hypothesis_medium"].tolist()

valid_refs = [r for r, h in zip(all_refs, all_hyps) if h]
valid_hyps = [h for h in all_hyps if h]
if valid_refs:
    _out = jiwer.process_words(valid_refs, valid_hyps)
    _corpus_errors = _out.substitutions + _out.deletions + _out.insertions
else:
    _corpus_errors = 0
# Empty hypotheses count as all-deletion errors
_corpus_errors += sum(len(r.split()) for r, h in zip(all_refs, all_hyps) if not h)
_total_ref_words = sum(len(r.split()) for r in all_refs)
overall_wer_medium = _corpus_errors / _total_ref_words if _total_ref_words else 0.0

print("\n" + "=" * 70)
print(f"WHISPER-MEDIUM RESULTS")
print(f"=" * 70)
print(f"Corpus-level WER (whisper-medium, test): {overall_wer_medium:.4f}  ({overall_wer_medium*100:.2f}%)")
print(f"Total samples processed: {len(df)}")

# --------------- Compare with Whisper Base ---------------
print("\n" + "=" * 70)
print("COMPARISON: whisper-base vs whisper-medium")
print("=" * 70)

if os.path.exists(BASE_RESULTS_CSV):
    df_base = pd.read_csv(BASE_RESULTS_CSV)

    # Corpus-level WER for base (including empty hypotheses)
    base_refs = df_base["reference"].tolist()
    base_hyps = df_base["hypothesis"].tolist()
    bv_refs = [r for r, h in zip(base_refs, base_hyps) if isinstance(h, str) and h]
    bv_hyps = [h for r, h in zip(base_refs, base_hyps) if isinstance(h, str) and h]
    if bv_refs:
        bv_out = jiwer.process_words(bv_refs, bv_hyps)
        bv_err = bv_out.substitutions + bv_out.deletions + bv_out.insertions
    else:
        bv_err = 0
    bv_err += sum(len(str(r).split()) for r, h in zip(base_refs, base_hyps) if not (isinstance(h, str) and h))
    bv_total = sum(len(str(r).split()) for r in base_refs)
    overall_wer_base = bv_err / bv_total if bv_total else float("nan")

    print(f"  Whisper-base   WER: {overall_wer_base:.4f}  ({overall_wer_base*100:.2f}%)")
    print(f"  Whisper-medium WER: {overall_wer_medium:.4f}  ({overall_wer_medium*100:.2f}%)")
    diff = overall_wer_base - overall_wer_medium
    print(f"  Improvement:        {diff:.4f}  ({diff*100:.2f} percentage points)")

    # Per-sample comparison (merge on ID)
    df_merged = df.merge(
        df_base[["ID", "wer"]].rename(columns={"wer": "wer_base"}),
        on="ID",
        how="left",
    )
    df_merged["wer_diff"] = df_merged["wer_base"] - df_merged["wer_medium"]
else:
    print(f"  [WARN] Base results not found at: {BASE_RESULTS_CSV}")
    print(f"         Run Task 1 first, then re-run this script.")
    print(f"\n  Whisper-medium WER: {overall_wer_medium:.4f}  ({overall_wer_medium*100:.2f}%)")

# --------------- Top 20 highest WER sentences ---------------
print("\n" + "=" * 70)
print("TOP 20 SENTENCES WITH HIGHEST WER (whisper-medium)")
print("=" * 70)

df_sorted = df.sort_values("wer_medium", ascending=False).head(20)

for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
    print(f"\n--- #{i} | ID: {row['ID']} | WER: {row['wer_medium']:.4f} ({row['wer_medium']*100:.1f}%) ---")
    print(f"  Split:        {row['split']}")
    dur = row['Speech_Duration_seconds']
    print(f"  Duration:     {float(dur):.2f}s" if dur != "" else "  Duration:     N/A")
    print(f"  Speech Class: {row['Speech_Class']}")
    print(f"  Region:       {row['Native_Region']}")
    print(f"  Gender:       {row['Gender']}")
    print(f"  Discipline:   {row['Discipline_Group']}")
    print(f"  Topic:        {row['Topic']}")
    print(f"  REFERENCE:    {row['reference'][:200]}")
    print(f"  HYPOTHESIS:   {row['hypothesis_medium'][:200]}")

# --------------- Analysis guidance ---------------
print("\n" + "=" * 70)
print("ANALYSIS GUIDANCE FOR HIGH-WER SAMPLES")
print("=" * 70)
print("""
Investigate the following factors for the top 20 high-WER samples:

1. SPEECH DURATION
   - Very short clips (<2s) give Whisper insufficient context.
   - Very long clips may cause hallucination or repetition.

2. SPEECH RATE (Speech_Class)
   - FAST speakers cause more recognition errors.
   - Check if high-WER samples cluster in FAST class.

3. SPEAKER ACCENT / REGION (Native_Region)
   - Indian English varies significantly by region (NORTH/SOUTH/EAST/WEST).
   - Whisper is trained mostly on Western English; regional accents increase WER.

4. DOMAIN-SPECIFIC TERMINOLOGY (Discipline_Group / Topic)
   - Technical terms (engineering jargon, acronyms) are often out-of-vocabulary.
   - Check if high-WER samples share specific disciplines or topics.

5. AUDIO QUALITY
   - Background noise, microphone issues, or overlapping speech.
   - Lecture recordings may have variable quality.

6. REFERENCE TRANSCRIPT ISSUES
   - Some references may themselves contain errors or inconsistencies.
   - Manual review of reference vs audio may reveal annotation issues.

7. CODE-SWITCHING / NON-ENGLISH WORDS
   - Speakers may mix Hindi or regional language words into English lectures.
   - Whisper (English mode) cannot handle these segments.

To investigate programmatically:
  - Group high-WER samples by Speech_Class, Native_Region, Discipline_Group
  - Plot WER distribution vs Speech_Duration_seconds
  - Compare WER across Gender, Region, and Discipline
  - Manually listen to a few high-WER audio samples
""")

# --------------- Save final results ---------------
output_csv = "wer_whisper_medium_results.csv"
df.to_csv(output_csv, index=False)
print(f"Full results saved to: {output_csv}")

# Clean up partial file
if os.path.exists("wer_whisper_medium_results_partial.csv"):
    os.unlink("wer_whisper_medium_results_partial.csv")

print("\nDone.")
