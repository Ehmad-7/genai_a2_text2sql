# Task 1: starter code report

## Pairs per split

| Split | Pairs |
|---|---|
| Train | 56,355 (56,336 kept after dropping 19 over-long pairs) |
| Dev | 8,421 |
| Test | 15,878 |

Mean and maximum lengths are in `table1_data.md`.

## Output of `check_starter.py`

```
train_pairs.jsonl: kept 56336, skipped 19
dev_pairs.jsonl: kept 8421, skipped 0
src (64, 94) tgt (64, 24)
encoder input (64, 94, 256) decoder input (64, 23, 256)
```

The batch lengths (94 and 24) change from run to run because the training loader shuffles. The decoder input is one position shorter than the target because it is `tgt[:, :-1]` (teacher forcing).

## First training example

```
src: what is terrence ross' nationality <sep> <c0> player <c1> no. <c2> nationality <c3> position <c4> years in toronto <c5> school/club team
tgt: select <c2> where <c0> = terrence ross
```

## Tokenizer check

- `<sep>` and the column tokens `<c0>` to `<c63>` are single tokens and are never split.
- `sp.decode(sp.encode(tgt)) == tgt` prints `True`.
- Every special symbol is preceded by a separate `▁` token (SentencePiece's word-start marker), so each column costs two tokens in the source and each generated `<cK>` costs two in the target.

## Figure 1: positional-encoding heat-map

File: `fig1_positional_encoding.png` (first 100 positions by 256 dimensions).

What it shows:
- Left side (dimensions 0 to about 80): fast oscillation, because these dimensions use high-frequency waves.
- Right side: very low frequencies, so each column barely changes over 100 positions. The red and pale stripes come from the sine/cosine interleaving (`sin` near 0 in even dimensions, `cos` near 1 in odd dimensions).
- Top row (position 0): `sin(0) = 0` and `cos(0) = 1`, so it alternates pale and red all the way across.

Two-sentence explanation (draft, rewrite in your own words before submitting):

> Each pair of dimensions is a sine and cosine wave with a different wavelength, from fast at low dimensions to very slow at high ones. Together they give every position a unique pattern, so the model can tell positions apart and judge relative distance even though attention itself has no notion of order.

## Possible viva questions for this part

- Why do the wavelengths form a geometric progression up to 10000?
- Why is the encoding added to the embedding rather than concatenated?
- Why are embeddings multiplied by the square root of d_model?
