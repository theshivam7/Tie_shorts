# Whisper Medium -- WER Summary (all 4 modes)

| mode | reference_source | corpus_wer | mean_wer | median_wer | std_wer | p90_wer | p95_wer | num_samples | num_empty_hyps | total_ref_words | total_errors |
|------|------------------|-----------:|---------:|-----------:|--------:|--------:|--------:|------------:|---------------:|----------------:|-------------:|
| raw | Transcript | 0.2419 | 0.2552 | 0.2143 | 0.2130 | 0.4348 | 0.5517 | 986 | 0 | 51755 | 12521 |
| normalized | Normalised_Transcript | 0.1923 | 0.2004 | 0.1552 | 0.2099 | 0.3784 | 0.4857 | 986 | 0 | 52331 | 10062 |
| double_normalized | Normalised_Transcript | 0.1568 | 0.1652 | 0.1200 | 0.1660 | 0.3265 | 0.4384 | 985 | 0 | 51798 | 8122 |
| whisper_normalized | Transcript | 0.1453 | 0.1575 | 0.1091 | 0.2035 | 0.3133 | 0.4151 | 986 | 0 | 51815 | 7528 |

## Notes

- **raw**: no normalization on either side
- **normalized**: reference as-is from `Normalised_Transcript`, hypothesis normalized with `EnglishTextNormalizer`
- **double_normalized**: both sides normalized; reference from `Normalised_Transcript`
- **whisper_normalized**: both sides normalized; reference from original `Transcript` (gold standard)
