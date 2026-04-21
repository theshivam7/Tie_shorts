# Whisper Large -- WER Summary (all 4 modes)

| mode | reference_source | corpus_wer | mean_wer | median_wer | std_wer | p90_wer | p95_wer | num_samples | num_empty_hyps | total_ref_words | total_errors |
|------|------------------|-----------:|---------:|-----------:|--------:|--------:|--------:|------------:|---------------:|----------------:|-------------:|
| raw | Transcript | 0.2591 | 0.2774 | 0.2174 | 0.2573 | 0.4815 | 0.6596 | 986 | 0 | 51755 | 13411 |
| normalized | Normalised_Transcript | 0.2067 | 0.2194 | 0.1579 | 0.2497 | 0.4237 | 0.5667 | 986 | 0 | 52331 | 10819 |
| double_normalized | Normalised_Transcript | 0.1710 | 0.1837 | 0.1228 | 0.2167 | 0.3696 | 0.5217 | 985 | 0 | 51798 | 8858 |
| whisper_normalized | Transcript | 0.1602 | 0.1769 | 0.1130 | 0.2465 | 0.3472 | 0.5106 | 986 | 0 | 51815 | 8300 |

## Notes

- **raw**: no normalization on either side
- **normalized**: reference as-is from `Normalised_Transcript`, hypothesis normalized with `EnglishTextNormalizer`
- **double_normalized**: both sides normalized; reference from `Normalised_Transcript`
- **whisper_normalized**: both sides normalized; reference from original `Transcript` (gold standard)
