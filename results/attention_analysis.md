# Cross-attention analysis (Task 5.3)

Figure: `fig4_attention_map.png`, made by `plot_attention.py`. It shows the decoder cross-attention of the last layer, averaged over the four heads, for one dev example: rows are generated tokens, columns are source tokens. The weights come from a clean forward pass over the model's own beam-search output.

The question from the brief: do the generated `<cK>` tokens attend to the matching column in the source?

## Three examples (dev examples 0, 1 and 4)

| Example | Generated | Peak on | Output correct? |
|---|---|---|---|
| 0 | `<c3>` | `▁position` (question word) | yes |
| 0 | `<c0>` | `<c5>` token | no (gold is `<c5>`) |
| 1 | `<c5>` | `▁schools` | yes |
| 1 | `<c1>` | `▁number` | yes |
| 4 | `<c0>` | `▁player` | yes |
| 4 | `<c2>` | `▁guard` (the value) | no (gold is `<c3>`) |
| 4 | `<c4>` | `▁toronto` | yes |

In none of the seven cases was the peak inside the matching column's span in the source.

## Statistics over 300 dev examples (`attention_stats.py`, greedy decoding)

| Measure | Result |
|---|---|
| Generated `<cK>` tokens analysed | 634 |
| Peak on a question token | 88.3% |
| Peak anywhere in the column list | 11.7% |
| Peak on its own `<cK>` token | 1.1% |
| Peak inside its own column span | 4.9% |
| Mean attention mass on own span | 7.4% |
| Mean attention mass on the question | 52.2% |

Raw output: `results/checks/attention_stats.txt`.

## Reading

- The expectation that `<cK>` tokens attend to the matching column is not supported in the last decoder layer, averaged over heads. The attention sits mostly on the question words that name the attribute ("position", "schools", "player").
- This does not mean the model ignores the columns: select accuracy is 91.58%, so column information is used somewhere.
- Possible explanation (not tested): encoder self-attention may already have mixed the column labels into the question-word representations, so the decoder can read the column index from the question words and does not need to look at the `<cK>` tokens. A test would be to plot encoder self-attention from question words to the column list, or to inspect earlier decoder layers and individual heads.

## Caveats

- Averaging over four heads can hide heads that behave differently.
- Only the last decoder layer was inspected.
- The statistics use the first 300 dev examples, not a random sample.
- A column span is only a few tokens out of about 42, so some attention mass lands there by chance. The 7.4% should be compared with the uniform-attention expectation before it is quoted as evidence either way (TODO).
- The span of a column starts at its `<cK>` token, so the `▁` marker in front of the next column counts toward the previous one.
