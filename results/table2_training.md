| | |
|---|---|
| Trainable parameters | 7,577,600 |
| Epochs trained / best epoch | 20 / 19 |
| Best dev loss | 1.4645 |
| Training time and GPU | about 21 min (about 64 s per epoch), Tesla T4 on Google Colab |

## Training configuration

| Setting | Value |
|---|---|
| d_model / heads / layers / d_ff | 256 / 4 / 3 + 3 / 1024 |
| Dropout | 0.1 |
| Normalisation | post-norm, `LayerNorm(x + Sublayer(x))` |
| Weight sharing | encoder embedding = decoder embedding = output projection |
| Loss | cross-entropy, label smoothing 0.1, ignoring `<pad>`, averaged per token |
| Optimiser | Adam, betas (0.9, 0.98), eps 1e-9 |
| LR schedule | `d_model^-0.5 * min(step^-0.5, step * 4000^-1.5)`, stepped every batch (peak 9.88e-4 at step 4000) |
| Batch size / epochs | 64 / 20 |
| Checkpoint | lowest dev loss (epoch 19), saved to `model_weights/best.pt` |

## Loss per epoch (label-smoothed, per token)

| Epoch | Train | Dev | LR at end of epoch |
|---|---|---|---|
| 1 | 4.3403 | 3.2128 | 2.18e-04 |
| 2 | 2.9041 | 2.4935 | 4.36e-04 |
| 3 | 2.3828 | 2.1981 | 6.53e-04 |
| 4 | 2.1003 | 1.9763 | 8.71e-04 |
| 5 | 1.9240 | 1.8381 | 9.42e-04 |
| 6 | 1.7625 | 1.6973 | 8.60e-04 |
| 7 | 1.6531 | 1.6371 | 7.96e-04 |
| 8 | 1.5819 | 1.5946 | 7.44e-04 |
| 9 | 1.5364 | 1.5584 | 7.02e-04 |
| 10 | 1.4994 | 1.5448 | 6.66e-04 |
| 11 | 1.4695 | 1.5184 | 6.35e-04 |
| 12 | 1.4484 | 1.5215 | 6.08e-04 |
| 13 | 1.4303 | 1.4980 | 5.84e-04 |
| 14 | 1.4148 | 1.4893 | 5.63e-04 |
| 15 | 1.4014 | 1.4891 | 5.44e-04 |
| 16 | 1.3906 | 1.4828 | 5.26e-04 |
| 17 | 1.3798 | 1.4771 | 5.11e-04 |
| 18 | 1.3720 | 1.4689 | 4.96e-04 |
| 19 | 1.3658 | 1.4645 | 4.83e-04 |
| 20 | 1.3585 | 1.4646 | 4.71e-04 |

## Notes

- With label smoothing 0.1 over an 8,000-token vocabulary the loss cannot go below about 1.2, so a final loss near 1.4 is close to the practical floor. A single batch memorised in the overfit test reached 1.26.
- Dev loss is below train loss in the early epochs because the train loss is measured with dropout on, and the dev loss is measured in eval mode.
- The gap between train and dev loss stays small (about 0.1), so there is no sign of heavy overfitting.

Figure: `fig2_loss_curves.png` (made by `plot_loss.py`).
