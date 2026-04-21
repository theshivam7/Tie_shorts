"""Normalization logic for 4-mode WER evaluation."""

from whisper.normalizers import EnglishTextNormalizer

whisper_normalizer = EnglishTextNormalizer()

MODES = ("raw", "normalized", "double_normalized", "whisper_normalized")


def get_ref_and_hyp(
    sample: dict,
    hyp_raw: str,
    mode: str,
) -> tuple[str, str]:
    """Return (reference, hypothesis) texts for the given evaluation mode.

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
        ref = transcript.strip()
        hyp = hyp_raw.strip()
    elif mode == "normalized":
        ref = normalised_transcript.strip()
        hyp = whisper_normalizer(hyp_raw) if hyp_raw.strip() else ""
    elif mode == "double_normalized":
        ref = whisper_normalizer(normalised_transcript) if normalised_transcript.strip() else ""
        hyp = whisper_normalizer(hyp_raw) if hyp_raw.strip() else ""
    elif mode == "whisper_normalized":
        ref = whisper_normalizer(transcript) if transcript.strip() else ""
        hyp = whisper_normalizer(hyp_raw) if hyp_raw.strip() else ""
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return ref, hyp
