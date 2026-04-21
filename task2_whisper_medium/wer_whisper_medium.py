"""
Task 2: Whisper Medium -- 4-mode WER evaluation.

Transcribes all audio using openai/whisper-medium and computes WER
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

from utils.normalize import MODES, get_ref_and_hyp, get_reference_source
from utils.transcribe import transcribe_sample
from utils.wer_compute import compute_corpus_wer, compute_sample_wer
from utils.io_helpers import (
    load_dataset_test,
    results_dir,
    save_mode_csv,
    save_checkpoint,
    remove_checkpoint,
    save_summary_csv,
)

warnings.filterwarnings("ignore")

MODEL_NAME = "medium"

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

mode_rows: dict[str, list[dict]] = {m: [] for m in MODES}

print(f"--- Processing test split ({len(ds)} samples) ---")

for sample in tqdm(ds, desc="test (transcribing)"):
    ref_raw_transcript = (sample.get("Transcript") or "").strip()
    if not ref_raw_transcript:
        continue

    sample_id = sample.get("ID", "")

    # Resume from checkpoint if this sample was already transcribed
    hyp_raw = None
    if str(sample_id) in completed_ids:
        ckpt_row = next((r for r in checkpoint_rows if str(r["ID"]) == str(sample_id)), None)
        if ckpt_row is not None:
            hyp_raw = str(ckpt_row.get("hypothesis_raw") or "")

    # Fall back to re-transcription if checkpoint lookup failed (prevents silent data loss)
    if hyp_raw is None:
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
        ref_raw, ref, hyp = get_ref_and_hyp(sample, hyp_raw, mode_name)

        if not ref:
            continue

        wer = compute_sample_wer(ref, hyp)

        mode_rows[mode_name].append({
            **metadata,
            "mode": mode_name,
            "reference_source": get_reference_source(mode_name),
            "reference_raw": ref_raw,
            "reference": ref,
            "hypothesis_raw": hyp_raw,
            "hypothesis": hyp,
            "wer": round(wer, 4),
        })

    checkpoint_rows.append({**metadata, "hypothesis_raw": hyp_raw})

    if len(checkpoint_rows) % 200 == 0:
        save_checkpoint(checkpoint_rows, MODEL_NAME)
        print(f"  [checkpoint] {len(checkpoint_rows)} samples saved")

# --------------- Save results, print summary, top 20 per mode ---------------
print("\n" + "=" * 70)
print(f"WHISPER-{MODEL_NAME.upper()} RESULTS (4-mode WER)")
print("=" * 70)

summary_rows: list[dict] = []

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
    print(f"    Reference source: {get_reference_source(mode_name)}")
    print(f"    Corpus WER: {stats['corpus_wer']:.4f}  ({stats['corpus_wer']*100:.2f}%)")
    print(f"    Mean WER:   {stats['mean_wer']:.4f}  |  Median WER: {stats['median_wer']:.4f}  |  Std: {stats['std_wer']:.4f}")
    print(f"    P90 WER:    {stats['p90_wer']:.4f}  |  P95 WER:    {stats['p95_wer']:.4f}")
    print(f"    Samples: {stats['num_samples']}  |  Empty hyps: {stats['num_empty_hyps']}")
    print(f"    Total ref words: {stats['total_ref_words']}  |  Total errors: {stats['total_errors']}")
    print(f"    Saved to: {out_path}")

    summary_rows.append({
        "model": MODEL_NAME,
        "mode": mode_name,
        "reference_source": get_reference_source(mode_name),
        "corpus_wer": round(stats["corpus_wer"], 4),
        "mean_wer": round(stats["mean_wer"], 4),
        "median_wer": round(stats["median_wer"], 4),
        "std_wer": round(stats["std_wer"], 4),
        "p90_wer": round(stats["p90_wer"], 4),
        "p95_wer": round(stats["p95_wer"], 4),
        "num_samples": stats["num_samples"],
        "num_empty_hyps": stats["num_empty_hyps"],
        "total_ref_words": stats["total_ref_words"],
        "total_errors": stats["total_errors"],
    })

    df_mode = pd.DataFrame(rows)
    df_sorted = df_mode.sort_values("wer", ascending=False).head(20)

    print(f"\n    TOP 20 HIGHEST WER [{mode_name}]")
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        dur = row["Speech_Duration_seconds"]
        dur_str = f"{float(dur):.2f}s" if dur != "" else "N/A"
        print(f"    #{i:2d} ID: {row['ID']}  WER: {row['wer']:.4f} ({row['wer']*100:.1f}%)  "
              f"Dur: {dur_str}  Region: {row['Native_Region']}  Class: {row['Speech_Class']}")
        print(f"         REF : {str(row['reference'])[:120]}")
        print(f"         HYP : {str(row['hypothesis'])[:120]}")

    top20_path = os.path.join(results_dir(), f"top_20_high_wer_{MODEL_NAME}_{mode_name}.csv")
    df_sorted.to_csv(top20_path, index=False)
    print(f"\n    Top 20 saved to: {top20_path}")

# --------------- Save per-model WER summary (CSV + Markdown) ---------------
if summary_rows:
    summary_csv = save_summary_csv(summary_rows, MODEL_NAME)
    print(f"\n  WER summary CSV saved to: {summary_csv}")

    summary_md = os.path.join(results_dir(), f"wer_summary_{MODEL_NAME}.md")
    with open(summary_md, "w") as f:
        f.write(f"# Whisper {MODEL_NAME.title()} -- WER Summary (all 4 modes)\n\n")
        f.write("| mode | reference_source | corpus_wer | mean_wer | median_wer | std_wer | p90_wer | p95_wer | num_samples | num_empty_hyps | total_ref_words | total_errors |\n")
        f.write("|------|------------------|-----------:|---------:|-----------:|--------:|--------:|--------:|------------:|---------------:|----------------:|-------------:|\n")
        for r in summary_rows:
            f.write(f"| {r['mode']} | {r['reference_source']} | {r['corpus_wer']:.4f} | {r['mean_wer']:.4f} | {r['median_wer']:.4f} | {r['std_wer']:.4f} | {r['p90_wer']:.4f} | {r['p95_wer']:.4f} | {r['num_samples']} | {r['num_empty_hyps']} | {r['total_ref_words']} | {r['total_errors']} |\n")
        f.write("\n## Notes\n\n")
        f.write("- **raw**: no normalization on either side\n")
        f.write("- **normalized**: reference as-is from `Normalised_Transcript`, hypothesis normalized with `EnglishTextNormalizer`\n")
        f.write("- **double_normalized**: both sides normalized; reference from `Normalised_Transcript`\n")
        f.write("- **whisper_normalized**: both sides normalized; reference from original `Transcript` (gold standard)\n")
    print(f"  WER summary Markdown saved to: {summary_md}")

remove_checkpoint(MODEL_NAME)
print("\nDone.")
