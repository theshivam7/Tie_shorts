"""
Task 1: Whisper Base -- 4-mode WER evaluation.

Transcribes all audio using openai/whisper-base and computes WER
under four normalization modes: raw, normalized, double_normalized,
whisper_normalized.

Uses only the 'test' split.
"""

import os
import sys
import warnings

import pandas as pd
import torch
import whisper
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.normalize import MODES, get_ref_and_hyp
from utils.transcribe import transcribe_sample
from utils.wer_compute import compute_corpus_wer, compute_sample_wer
from utils.io_helpers import (
    load_dataset_test,
    results_dir,
    save_mode_csv,
    save_checkpoint,
    remove_checkpoint,
)

warnings.filterwarnings("ignore")

MODEL_NAME = "base"

# --------------- Load model ---------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading whisper-{MODEL_NAME} on device: {device} ...")
model = whisper.load_model(MODEL_NAME, device=device)
print("Model loaded.\n")

transcribe_kw = {"language": "en"}
if device == "cpu":
    transcribe_kw["fp16"] = False

# --------------- Load dataset ---------------
ds = load_dataset_test()

# --------------- Resume from checkpoint if available ---------------
checkpoint_path = os.path.join(results_dir(), f"wer_{MODEL_NAME}_partial.csv")
completed_ids: set[str] = set()
checkpoint_rows: list[dict] = []

if os.path.exists(checkpoint_path):
    df_partial = pd.read_csv(checkpoint_path)
    completed_ids = set(df_partial["ID"].astype(str).tolist())
    checkpoint_rows = df_partial.to_dict("records")
    print(f"  Resuming from checkpoint: {len(completed_ids)} samples already done\n")

# Per-mode result rows: { mode: [row_dicts] }
mode_rows: dict[str, list[dict]] = {m: [] for m in MODES}

print(f"--- Processing test split ({len(ds)} samples) ---")

for sample in tqdm(ds, desc="test (transcribing)"):
    ref_raw = sample.get("Transcript") or ""
    if not ref_raw.strip():
        continue

    sample_id = sample.get("ID", "")

    # Skip already-completed samples (resume support)
    if str(sample_id) in completed_ids:
        # Reconstruct mode rows from checkpoint
        ckpt_row = next((r for r in checkpoint_rows if str(r["ID"]) == str(sample_id)), None)
        if ckpt_row:
            hyp_raw = ckpt_row.get("hypothesis_raw", "")
            metadata = {
                "split": "test",
                "ID": sample_id,
                "Speaker_ID": sample.get("Speaker_ID", ""),
                "Gender": sample.get("Gender", ""),
                "Speech_Class": sample.get("Speech_Class", ""),
                "Native_Region": sample.get("Native_Region", ""),
                "Speech_Duration_seconds": sample.get("Speech_Duration_seconds") or "",
                "Discipline_Group": sample.get("Discipline_Group", ""),
                "Topic": sample.get("Topic", ""),
            }
            for mode_name in MODES:
                ref, hyp = get_ref_and_hyp(sample, hyp_raw, mode_name)
                if not ref:
                    continue
                wer = compute_sample_wer(ref, hyp)
                mode_rows[mode_name].append({
                    **metadata,
                    "reference": ref,
                    "hypothesis": hyp,
                    "hypothesis_raw": hyp_raw,
                    "wer": round(wer, 4),
                })
        continue

    hyp_raw = transcribe_sample(model, sample, transcribe_kw)

    metadata = {
        "split": "test",
        "ID": sample_id,
        "Speaker_ID": sample.get("Speaker_ID", ""),
        "Gender": sample.get("Gender", ""),
        "Speech_Class": sample.get("Speech_Class", ""),
        "Native_Region": sample.get("Native_Region", ""),
        "Speech_Duration_seconds": sample.get("Speech_Duration_seconds") or "",
        "Discipline_Group": sample.get("Discipline_Group", ""),
        "Topic": sample.get("Topic", ""),
    }

    for mode_name in MODES:
        ref, hyp = get_ref_and_hyp(sample, hyp_raw, mode_name)

        if not ref:
            continue

        wer = compute_sample_wer(ref, hyp)

        mode_rows[mode_name].append({
            **metadata,
            "reference": ref,
            "hypothesis": hyp,
            "hypothesis_raw": hyp_raw,
            "wer": round(wer, 4),
        })

    checkpoint_rows.append({**metadata, "hypothesis_raw": hyp_raw})

    if len(checkpoint_rows) % 200 == 0:
        save_checkpoint(checkpoint_rows, MODEL_NAME)
        print(f"  [checkpoint] {len(checkpoint_rows)} samples saved")

# --------------- Save results and print summary ---------------
print("\n" + "=" * 70)
print(f"WHISPER-{MODEL_NAME.upper()} RESULTS (4-mode WER)")
print("=" * 70)

for mode_name in MODES:
    rows = mode_rows[mode_name]
    if not rows:
        print(f"\n  [{mode_name}] No valid samples.")
        continue

    refs = [r["reference"] for r in rows]
    hyps = [r["hypothesis"] for r in rows]
    wers = [r["wer"] for r in rows]
    stats = compute_corpus_wer(refs, hyps, per_sample_wers=wers)

    out_path = save_mode_csv(rows, MODEL_NAME, mode_name)
    print(f"\n  [{mode_name}]")
    print(f"    Corpus WER: {stats['corpus_wer']:.4f}  ({stats['corpus_wer']*100:.2f}%)")
    print(f"    Mean WER:   {stats['mean_wer']:.4f}  |  Median WER: {stats['median_wer']:.4f}  |  Std: {stats['std_wer']:.4f}")
    print(f"    P90 WER:    {stats['p90_wer']:.4f}  |  P95 WER:    {stats['p95_wer']:.4f}")
    print(f"    Samples: {stats['num_samples']}  |  Empty hyps: {stats['num_empty_hyps']}")
    print(f"    Total ref words: {stats['total_ref_words']}  |  Total errors: {stats['total_errors']}")
    print(f"    Saved to: {out_path}")

# --------------- Top 20 highest WER sentences (whisper_normalized mode) ---------------
PRIMARY_MODE = "whisper_normalized"
primary_rows = mode_rows.get(PRIMARY_MODE, [])

if primary_rows:
    df_primary = pd.DataFrame(primary_rows)
    df_sorted = df_primary.sort_values("wer", ascending=False).head(20)

    print(f"\n{'=' * 70}")
    print(f"TOP 20 SENTENCES WITH HIGHEST WER (mode: {PRIMARY_MODE})")
    print("=" * 70)

    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"\n--- #{i} | ID: {row['ID']} | WER: {row['wer']:.4f} ({row['wer']*100:.1f}%) ---")
        dur = row["Speech_Duration_seconds"]
        print(f"  Duration:     {float(dur):.2f}s" if dur != "" else "  Duration:     N/A")
        print(f"  Speech Class: {row['Speech_Class']}")
        print(f"  Region:       {row['Native_Region']}")
        print(f"  Gender:       {row['Gender']}")
        print(f"  REFERENCE:    {str(row['reference'])[:200]}")
        print(f"  HYPOTHESIS:   {str(row['hypothesis'])[:200]}")

    top20_path = os.path.join(results_dir(), f"top_20_high_wer_{MODEL_NAME}.csv")
    df_sorted.to_csv(top20_path, index=False)
    print(f"\n  Top 20 saved to: {top20_path}")

remove_checkpoint(MODEL_NAME)
print("\nDone.")
