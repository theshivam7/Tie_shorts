# WER Evaluation Summary -- TIE_shorts (Indian English)

## Corpus-level WER (%) by Model and Evaluation Mode

| model | raw | normalized | double_normalized | whisper_normalized |
| --- | --- | --- | --- | --- |
| base | 27.86 | 21.32 | 17.88 | 16.92 |
| medium | 24.19 | 19.23 | 15.68 | 14.53 |
| large | 25.91 | 20.67 | 17.1 | 16.02 |

## Evaluation Modes

| Mode | reference_source | Reference normalization | Hypothesis normalization | Symmetric? |
|------|-----------------|------------------------|--------------------------|------------|
| raw | Transcript | None | None | Yes |
| normalized | Normalised_Transcript | None (as-is) | EnglishTextNormalizer | No |
| double_normalized | Normalised_Transcript | EnglishTextNormalizer | EnglishTextNormalizer | Yes |
| whisper_normalized | Transcript | EnglishTextNormalizer | EnglishTextNormalizer | Yes |

## Normalization Notes

- **raw**: No normalization on either side. WER is highest because punctuation and casing differences are penalized.
- **normalized**: Intentionally asymmetric — reference uses the dataset's `Normalised_Transcript` as-is while hypothesis gets `EnglishTextNormalizer`. This mode shows the impact of the dataset's own normalization quality on WER.
- **double_normalized**: Both sides get `EnglishTextNormalizer` applied to `Normalised_Transcript`. Fixes punctuation/casing issues but the dataset's bad number conversions (e.g. 'second' -> 'one s t') still affect the reference.
- **whisper_normalized**: Gold standard. Both sides apply `EnglishTextNormalizer` to the original `Transcript`, completely bypassing `Normalised_Transcript` errors. Expected to give the lowest WER.

## Column Schema

Each result CSV contains: `split, ID, Speaker_ID, Gender, Speech_Class, Native_Region, Speech_Duration_seconds, Discipline_Group, Topic, mode, reference_source, reference_raw, reference, hypothesis_raw, hypothesis, wer`

- `reference_raw`: source text before normalization (for verification)
- `reference`: text used for WER computation (after normalization)
- `hypothesis_raw`: raw Whisper output before normalization
- `hypothesis`: Whisper output after normalization
- In `raw` mode: `reference_raw == reference` and `hypothesis_raw == hypothesis`

## Best Model per Mode

- **raw**: Whisper medium (24.19%)
- **normalized**: Whisper medium (19.23%)
- **double_normalized**: Whisper medium (15.68%)
- **whisper_normalized**: Whisper medium (14.53%)
