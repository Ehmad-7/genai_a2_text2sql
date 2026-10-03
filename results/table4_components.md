| Component (dev, beam search k=4) | Correct (%) |
|---|---|
| sel column correct | 91.58 |
| agg correct | 89.89 |
| WHERE clause correct | 74.49 |

WHERE is correct when the set of conditions (column, operator, value) matches the gold set, ignoring order. Parse failures count as wrong on all three components. Gold round-trip ceiling: sel 100.00, agg 100.00, WHERE 99.49.
