"""
Stage 1: YouTube Closed Captions Fetch.

Fetches auto-generated English captions from YouTube for all test IDs.
Video IDs in the TIE_shorts dataset ARE YouTube video IDs.

Saves to results/stage1_raw_transcripts/wer_youtube_raw.csv.

No GPU needed. Run once; re-run normalize_and_score.py for WER.

Caption types fetched (in priority order):
  1. Manual English captions (most accurate)
  2. Auto-generated English captions (Google ASR — direct Whisper comparison)

Samples with no captions available are saved with empty hypothesis_raw
and flagged in the caption_type column as "unavailable".
"""

import os
import sys
import time
import warnings

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io_helpers import load_dataset_test, stage1_raw_dir

warnings.filterwarnings("ignore")

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
    )
    YTT_AVAILABLE = True
except ImportError:
    print("ERROR: youtube-transcript-api not installed.")
    print("Run: pip install youtube-transcript-api")
    sys.exit(1)

# --------------- Config ---------------
DELAY_BETWEEN_REQUESTS = 1.0  # seconds — avoids IP rate limiting
RETRY_ATTEMPTS = 2
MODEL_NAME = "youtube"

api = YouTubeTranscriptApi()


def fetch_caption(video_id: str) -> tuple[str, str]:
    """Fetch English captions for a YouTube video ID.

    Returns (text, caption_type) where caption_type is one of:
        "manual"       — human-created captions
        "auto"         — YouTube auto-generated captions (Google ASR)
        "unavailable"  — no English captions found
        "error"        — fetch failed (network issue, IP block, etc.)
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            transcript_list = api.list(video_id)

            # Try manual English first, then auto-generated
            for generated in [False, True]:
                try:
                    transcript = transcript_list.find_transcript(["en"])
                    if transcript.is_generated != generated and not generated:
                        continue
                    data = transcript.fetch()
                    snippets = list(data)
                    text = " ".join(s.text for s in snippets).strip()
                    kind = "auto" if transcript.is_generated else "manual"
                    return text, kind
                except Exception:
                    continue

            # Fallback: any English variant
            try:
                data = api.fetch(video_id)
                snippets = list(data)
                text = " ".join(s.text for s in snippets).strip()
                return text, "auto"
            except Exception:
                pass

            return "", "unavailable"

        except (TranscriptsDisabled, NoTranscriptFound):
            return "", "unavailable"
        except VideoUnavailable:
            return "", "unavailable"
        except Exception as e:
            err = str(e)
            if "IpBlocked" in err or "ip" in err.lower():
                print(f"\n  [WARN] IP blocked — waiting 30s before retry {attempt+1}/{RETRY_ATTEMPTS}")
                time.sleep(30)
                continue
            if attempt == RETRY_ATTEMPTS - 1:
                return "", "error"
            time.sleep(5)

    return "", "error"


# --------------- Load dataset ---------------
ds = load_dataset_test()

# --------------- Resume from checkpoint ---------------
from utils.io_helpers import results_dir
checkpoint_path = os.path.join(results_dir(), f"wer_{MODEL_NAME}_partial.csv")
completed_ids: set[str] = set()
checkpoint_rows: list[dict] = []

if os.path.exists(checkpoint_path):
    df_partial = pd.read_csv(checkpoint_path)
    completed_ids = set(df_partial["ID"].astype(str).tolist())
    checkpoint_rows = df_partial.to_dict("records")
    print(f"  Resuming from checkpoint: {len(completed_ids)} samples already done\n")

all_rows: list[dict] = []
stats = {"manual": 0, "auto": 0, "unavailable": 0, "error": 0}

print(f"--- Fetching YouTube captions ({len(ds)} samples) ---")
print(f"    Delay between requests: {DELAY_BETWEEN_REQUESTS}s\n")

for sample in tqdm(ds, desc="fetching captions"):
    transcript = (sample.get("Transcript") or "").strip()
    if not transcript:
        continue

    sample_id = sample.get("ID", "")

    # Resume from checkpoint
    if str(sample_id) in completed_ids:
        ckpt_row = next((r for r in checkpoint_rows if str(r["ID"]) == str(sample_id)), None)
        if ckpt_row is not None:
            all_rows.append(ckpt_row)
            continue

    hyp_raw, caption_type = fetch_caption(sample_id)
    stats[caption_type] = stats.get(caption_type, 0) + 1

    row = {
        "split": "test",
        "ID": sample_id,
        "Speaker_ID": sample.get("Speaker_ID", ""),
        "Gender": sample.get("Gender", ""),
        "Speech_Class": sample.get("Speech_Class", ""),
        "Native_Region": sample.get("Native_Region", ""),
        "Speech_Duration_seconds": sample.get("Speech_Duration_seconds") or "",
        "Discipline_Group": sample.get("Discipline_Group", ""),
        "Topic": sample.get("Topic", ""),
        "transcript_raw": transcript,
        "normalised_transcript_raw": str(sample.get("Normalised_Transcript") or "").strip(),
        "hypothesis_raw": hyp_raw,
        "caption_type": caption_type,
    }

    all_rows.append(row)
    checkpoint_rows.append(row)

    # Checkpoint every 50 samples
    if len(all_rows) % 50 == 0:
        pd.DataFrame(checkpoint_rows).to_csv(checkpoint_path, index=False)
        tqdm.write(f"  [checkpoint] {len(all_rows)} done — manual:{stats['manual']} auto:{stats['auto']} unavailable:{stats['unavailable']}")

    time.sleep(DELAY_BETWEEN_REQUESTS)

# --------------- Save ---------------
out_path = os.path.join(stage1_raw_dir(), f"wer_{MODEL_NAME}_raw.csv")
pd.DataFrame(all_rows).to_csv(out_path, index=False)

available = stats["manual"] + stats["auto"]
print(f"\nSaved: {out_path}")
print(f"Total: {len(all_rows)} samples")
print(f"  Manual captions : {stats['manual']}")
print(f"  Auto captions   : {stats['auto']}")
print(f"  Unavailable     : {stats['unavailable']}")
print(f"  Errors          : {stats['error']}")
print(f"  Coverage        : {available}/{len(all_rows)} ({available/len(all_rows)*100:.1f}%)")
print("\nRun 'python normalize_and_score.py' for WER evaluation.")

if os.path.exists(checkpoint_path):
    os.unlink(checkpoint_path)
print("\nDone.")
