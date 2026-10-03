# Correctness checks (Section 3.2 of the brief)

Raw outputs live in `results/checks/`. Regenerate the unit-test outputs from the project root:

```bash
mkdir -p results/checks
python test_attention.py > results/checks/attention.txt
python test_layers.py    > results/checks/layers.txt
python test_masks.py     > results/checks/masks.txt
python test_model.py     > results/checks/model.txt
python test_parser.py    > results/checks/parser.txt
python test_decode.py    > results/checks/decode.txt
```

| Check | Test | Result |
|---|---|---|
| Causal mask: changing the last decoder-input token leaves all earlier outputs unchanged | `test_model.py`, also `test_layers.py` (decoder) | Passed |
| Padding mask: adding `<pad>` tokens to a source leaves the output unchanged | `test_model.py`, also `test_layers.py` (encoder, decoder cross-attention) | Passed |
| Attention rows sum to 1 over the unmasked positions | `test_attention.py` | TODO: paste the output of `results/checks/attention.txt` |
| Weight sharing: output projection weight `is` the embedding weight | `test_model.py` | Passed |
| Encoder and decoder share one embedding object | `test_model.py` | Passed |
| Learning-rate schedule rises linearly for 4000 steps, then decays as step^-0.5 | `plot_lr.py` | Peak 0.000988 at step 4000, see `fig3_lr_schedule.png` |
| Gold round-trip: gold dev targets through the parser and the official evaluator, execution accuracy above 99% | `gold_roundtrip.py` | Passed: 99.49% execution accuracy, 0 parse failures over 8421 lines (`checks/gold_roundtrip.txt`) |

## Model-level numbers from `test_model.py`

```
Trainable parameters: 7577600
initial loss: 9.67   (varies slightly between runs; random token ids)
Transformer test passed
```

Parameter breakdown:

| Part | Parameters |
|---|---|
| Shared embedding (8000 x 256) | 2,048,000 |
| Encoder layer (attention 263,168 + FFN 525,568 + 2 LayerNorm 1,024) | 789,760 |
| 3 encoder layers | 2,369,280 |
| Decoder layer (2 attentions 526,336 + FFN 525,568 + 3 LayerNorm 1,536) | 1,053,440 |
| 3 decoder layers | 3,160,320 |
| Output projection | 0 extra (tied to the embedding) |
| Total | 7,577,600 |

## Decoding checks (`test_decode.py`)

| Check | Result |
|---|---|
| Batched greedy decoding equals decoding each example alone (padding does not change the output) | Passed |
| Beam search with beam size 1 equals greedy decoding | Passed |
| Beam search with beam size 4 returns a valid-length sequence | Passed |

## Parser checks (`test_parser.py`)

Includes values that contain the words "and" and "where" (`tom and jerry`, `where is it`), which must not split a condition, and malformed strings, which must return `None` and never raise. All passed.

## Training-loop sanity checks

| Check | Result |
|---|---|
| Overfit one batch of 64 examples (fixed lr 1e-3, 200 steps) | Loss per token 4.46 at step 20, 1.26 at step 200. The floor under label smoothing 0.1 is about 1.2. |
| Smoke test, 2 epochs of 5 batches | Ran, wrote the log and both checkpoints |
| Resume from `last.pt` | Resumed at epoch 3 with lr 3.95e-06, which equals `noam_lr(16)`, so the scheduler state was restored |
| Full run | 20 epochs on a Colab T4, best dev loss 1.4645 at epoch 19 (`table2_training.md`) |

## Warning-sign thresholds from the brief

| Warning sign | Threshold | This project |
|---|---|---|
| Possible bug | execution accuracy below 20% | 70.98% (dev, beam) |
| Decoding bug | parse failures above 5% | 0.26% (dev, beam), 0.31% (test) |
| Gold leaking | execution accuracy above 95% | no |
