# Whisper Base -- WER Summary (all 4 modes)

| mode | reference_source | corpus_wer | mean_wer | median_wer | std_wer | p90_wer | p95_wer | num_samples | num_empty_hyps | total_ref_words | total_errors |
|------|------------------|-----------:|---------:|-----------:|--------:|--------:|--------:|------------:|---------------:|----------------:|-------------:|
| raw | Transcript | 0.2786 | 0.2935 | 0.2459 | 0.2213 | 0.4933 | 0.6184 | 986 | 0 | 51755 | 14421 |
| normalized | Normalised_Transcript | 0.2132 | 0.2230 | 0.1786 | 0.2141 | 0.4255 | 0.5500 | 986 | 0 | 52331 | 11155 |
| double_normalized | Normalised_Transcript | 0.1788 | 0.1891 | 0.1400 | 0.1746 | 0.3810 | 0.5111 | 985 | 0 | 51798 | 9260 |
| whisper_normalized | Transcript | 0.1692 | 0.1837 | 0.1321 | 0.2111 | 0.3718 | 0.5000 | 986 | 0 | 51815 | 8769 |

## Notes

- **raw**: no normalization on either side
- **normalized**: reference as-is from `Normalised_Transcript`, hypothesis normalized with `EnglishTextNormalizer`
- **double_normalized**: both sides normalized; reference from `Normalised_Transcript`
- **whisper_normalized**: both sides normalized; reference from original `Transcript` (gold standard)
