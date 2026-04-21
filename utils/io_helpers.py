"""Dataset loading and CSV I/O utilities."""

import os

import pandas as pd
from datasets import load_dataset

HF_CACHE = os.path.join(os.path.expanduser("~"), "hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)
os.environ.setdefault("HF_DATASETS_CACHE", HF_CACHE)


def load_dataset_test():
    """Load raianand/TIE_shorts test split."""
    print("Loading dataset raianand/TIE_shorts (test split) ...")
    print(f"  Cache directory: {HF_CACHE}")
    ds = load_dataset("raianand/TIE_shorts", split="test", cache_dir=HF_CACHE)
    print(f"  Loaded {len(ds)} samples\n")
    return ds


def results_dir() -> str:
    """Return the project-level results directory."""
    return os.path.join(os.path.dirname(__file__), "..", "results")


def save_mode_csv(
    rows: list[dict],
    model_name: str,
    mode: str,
) -> str:
    """Save per-sample results CSV for a given model and mode.

    Returns the output file path.
    """
    out_dir = os.path.join(results_dir(), mode)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"wer_{model_name}_{mode}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def save_checkpoint(rows: list[dict], model_name: str) -> str:
    """Save a partial checkpoint CSV for crash recovery."""
    out_path = os.path.join(results_dir(), f"wer_{model_name}_partial.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def remove_checkpoint(model_name: str) -> None:
    """Remove partial checkpoint CSV after successful completion."""
    out_path = os.path.join(results_dir(), f"wer_{model_name}_partial.csv")
    if os.path.exists(out_path):
        os.unlink(out_path)
