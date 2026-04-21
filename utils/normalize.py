"""Normalization logic for 4-mode WER evaluation."""

from whisper.normalizers import EnglishTextNormalizer

whisper_normalizer = EnglishTextNormalizer()

MODES = ("raw", "normalized", "double_normalized", "whisper_normalized")

_REFERENCE_SOURCE = {
    "raw": "Transcript",
    "normalized": "Normalised_Transcript",
    "double_normalized": "Normalised_Transcript",
    "whisper_normalized": "Transcript",
}


def get_reference_source(mode: str) -> str:
    """Return the dataset column used as reference for the given mode."""
    if mode not in _REFERENCE_SOURCE:
        raise ValueError(f"Unknown mode: {mode}")
    return _REFERENCE_SOURCE[mode]


def get_ref_and_hyp(
    sample: dict,
    hyp_raw: str,
    mode: str,
) -> tuple[str, str, str]:
    """Return (reference_raw, reference, hypothesis) for the given evaluation mode.

    reference_raw -- source text before any normalization (for verification)
    reference     -- text used for WER computation (after normalization)
    hypothesis    -- Whisper output after normalization

    Modes:
        raw               -- Transcript as-is vs Whisper output as-is
        normalized        -- Normalised_Transcript as-is vs EnglishTextNormalizer(hyp)
        double_normalized -- EnglishTextNormalizer(Normalised_Transcript) vs EnglishTextNormalizer(hyp)
        whisper_normalized -- EnglishTextNormalizer(Transcript) vs EnglishTextNormalizer(hyp)
    """
    def _safe_str(val) -> str:
        if val is None or (isinstance(val, float) and val != val):
            return ""
        return str(val)

    transcript = _safe_str(sample.get("Transcript"))
    normalised_transcript = _safe_str(sample.get("Normalised_Transcript"))
    hyp_raw = _safe_str(hyp_raw)

    if mode == "raw":
        ref_raw = transcript.strip()
        ref = ref_raw
        hyp = hyp_raw.strip()
    elif mode == "normalized":
        ref_raw = normalised_transcript.strip()
        ref = ref_raw
        hyp = whisper_normalizer(hyp_raw) if hyp_raw.strip() else ""
    elif mode == "double_normalized":
        ref_raw = normalised_transcript.strip()
        ref = whisper_normalizer(normalised_transcript) if normalised_transcript.strip() else ""
        hyp = whisper_normalizer(hyp_raw) if hyp_raw.strip() else ""
    elif mode == "whisper_normalized":
        ref_raw = transcript.strip()
        ref = whisper_normalizer(transcript) if transcript.strip() else ""
        hyp = whisper_normalizer(hyp_raw) if hyp_raw.strip() else ""
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return ref_raw, ref, hyp
