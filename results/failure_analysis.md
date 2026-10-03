# Failure analysis (dev, beam search k=4)

Produced by `make_samples.py`. Logical-form correct: 5402 of 8421 (64.15%, the same number the official evaluator reports).

| Failure | Count | % of dev |
|---|---|---|
| Wrong aggregation | 751 | 8.9 |
| Wrong select column | 687 | 8.2 |
| Wrong column in WHERE | 687 | 8.2 |
| Wrong value | 516 | 6.1 |
| Missing condition | 163 | 1.9 |
| Extra condition | 145 | 1.7 |
| Wrong operator | 48 | 0.6 |
| Parse failure | 22 | 0.3 |
| **Total wrong** | **3019** | **35.9** |

How to read it:

- Each wrong example is counted once, under its first failing component, checked in this order: parse, select column, aggregation, WHERE. So "wrong aggregation" means the select column was right and the aggregation was wrong. A question with both wrong is counted under select only.
- These categories come from comparing logical forms. The official execution accuracy is 6.8 points higher than logical-form accuracy, so some of the "wrong value" cases probably return the right answer when executed (not verified).
- Aggregation is the largest bucket, which agrees with Table 4 (agg 89.89% against sel 91.58%): the model has to infer "how many", "highest" or "total" from wording that is often implicit.
- The WHERE clause is the weakest component overall (Table 4: 74.49%). Its failures are split between the wrong column, a wrong value (for example a copy error such as `1996-97` becoming `1996-77`), and missing or extra conditions.

Examples with the failure named are in `samples.md`.
