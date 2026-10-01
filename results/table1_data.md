| | Train | Dev | Test |
|----|-------|-----|------|
| Pairs | 56355 | 8421 | 15878 |
| Mean / Max source length(tokens) | 42.5 / 222 | 42.5 / 167 | 42.7 / 260 |
| Mean / max target length(tokens) | 14.8 / 65 | 14.8 / 44 | 14.9 / 46 |
| Pairs dropped as too long | 19 | - | - |

Lengths include </s> on the source and <s> and </s> on the target, and are computed over all pairs before dropping (train pairs are dropped if source > 160 or target > 64).
