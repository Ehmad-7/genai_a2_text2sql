| Split | Decoding | Logical form (%) | Execution (%) | Parse failures (%) |
|---|---|---|---|---|
| Dev | greedy | 63.42 | 70.18 | 0.59 |
| Dev | beam (4) | 64.15 | 70.98 | 0.26 |
| Test | beam (4) | 64.14 | 70.41 | 0.31 |

Scores come from the official WikiSQL evaluator. Parse failures are the lines written as `{"error": "parse"}`: 50 of 8421 (dev greedy), 22 of 8421 (dev beam), 49 of 15878 (test).

Beam search (k=4) was chosen for the single test run from the dev results: it was better than greedy on all three dev metrics (+0.8 execution points). The test split was decoded once.

Reference points:
- The original LSTM sequence-to-sequence baseline in the WikiSQL paper reached about 36% execution accuracy on test.
- Gold round-trip ceiling: feeding the gold dev targets through the tokenizer, the parser and the official evaluator gives 99.49% execution accuracy, so no score can reach 100% under this framing (lowercasing and tokenizer normalisation change a few gold values).
- Test execution accuracy is within 0.6 points of dev, so selecting the decoding method on dev did not overfit to dev.

Raw evaluator output: `results/checks/dev_greedy_eval.txt`, `dev_beam_eval.txt`, `test_eval.txt`.
