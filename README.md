# Text-to-SQL with a Transformer built from scratch

Generative AI, Assignment 02 (Fall 2026). An English question plus the column names of one table goes in, and a SQL query comes out. The model is the original encoder-decoder Transformer from *Attention Is All You Need*, written in plain PyTorch layers (no `nn.Transformer`, no `nn.MultiheadAttention`, no Hugging Face, no pretrained weights) and trained once from random initialisation on WikiSQL.

```
Columns : Player, No., Nationality, Position, Years in Toronto, School/Club Team
Question: What is Terrence Ross' nationality?
SQL     : SELECT Nationality FROM table WHERE Player = 'terrence ross'
```

## Headline results

| Split | Decoding | Logical form (%) | Execution (%) | Parse failures (%) |
|---|---|---|---|---|
| Dev | greedy | 63.42 | 70.18 | 0.59 |
| Dev | beam (4) | 64.15 | 70.98 | 0.26 |
| Test | beam (4) | 64.14 | 70.41 | 0.31 |

Scores are from the official WikiSQL evaluator. The original LSTM sequence-to-sequence baseline reaches about 36% execution accuracy on test. Beam search (k=4) was chosen from the dev results and the test split was decoded once. Details in `results/table3_metrics.md`.

## How the task is framed

Columns are referred to by special tokens `<c0>`, `<c1>`, ..., so the model only has to point at a column and never has to spell its name.

```
source: what is terrence ross' nationality <sep> <c0> player <c1> no. <c2> nationality ...
target: select <c2> where <c0> = terrence ross
```

Source and target share one 8,000-piece BPE vocabulary, so the encoder embedding, the decoder embedding and the output projection share one weight matrix. The decoded string is parsed back into WikiSQL's format (`{"sel", "agg", "conds"}`) and then into readable SQL with the real column names.

## Model

| | |
|---|---|
| d_model | 256 |
| Heads | 4 (d_k = d_v = 64) |
| Layers | 3 encoder + 3 decoder |
| Feed-forward size | 1024 |
| Dropout | 0.1 |
| Normalisation | post-norm |
| Weight sharing | encoder embedding = decoder embedding = output projection |
| Trainable parameters | 7,577,600 |

Training: Adam (betas 0.9 and 0.98, eps 1e-9), the paper's warmup schedule with 4000 warmup steps, label smoothing 0.1, batch size 64, 20 epochs on a Colab T4 (about 21 minutes). The checkpoint with the lowest dev loss (epoch 19, dev loss 1.4645) is used for everything. See `results/table2_training.md`.

## Design choices worth knowing

- **Masks:** `True` means "hide" everywhere. The padding mask has shape `(B, 1, 1, L)`, the causal mask `(1, 1, T, T)`, and the decoder self-attention mask is their logical OR. Masked scores are filled with -1e9 before the softmax.
- **Initialisation:** the shared embedding is initialised with std `d_model ** -0.5` (with the pad row zeroed) and the other matrices with Xavier uniform. The embedding is multiplied by the square root of d_model, and with the output layer tied to it a unit-variance embedding would give a huge starting loss.
- **Loss:** summed over non-pad tokens and averaged per token, for both train and dev, so the two curves are comparable.
- **Beam search:** raw summed log-probability with no length normalisation, and the search stops once four hypotheses have finished.
- **Parser:** a new condition starts only at the pattern `and <cN> <op>`, so values that contain the words "and" or "where" are not split.

## Results and analysis

All of these are in `results/`.

| File | Content |
|---|---|
| `table1_data.md` | Table 1: data statistics |
| `table2_training.md` | Table 2: model and training, loss per epoch |
| `table3_metrics.md` | Table 3: official metrics |
| `table4_components.md` | Table 4: component accuracy (dev, beam) |
| `samples.md` | Ten dev examples, five right and five wrong, with the failure named |
| `failure_analysis.md` | Failure counts over all of dev |
| `attention_analysis.md` | Cross-attention findings |
| `correctness_checks.md` | All correctness checks |
| `task1_notes.md` | Task 1 report |

Table 4 (dev, beam search k=4):

| Component | Correct (%) |
|---|---|
| sel column | 91.58 |
| agg | 89.89 |
| WHERE clause | 74.49 |

Figures:

1. `fig1_positional_encoding.png`: positional-encoding heat-map
2. `fig2_loss_curves.png`: training and dev loss per epoch
3. `fig3_lr_schedule.png`: learning-rate schedule
4. `fig4_attention_map.png`: cross-attention map for one dev example
5. `fig5_frontend.png`: front-end screenshot (TODO)

In the last decoder layer, averaged over heads, the generated `<cK>` tokens mostly attend to the question words that name the attribute and not to the matching column in the source (peak inside the own column span in 4.9% of 634 cases). `results/attention_analysis.md` gives the numbers and the caveats.

## Repository layout

```
starter/               the given starter files and check_starter.py (unchanged), plus sql_sp.model
model/attention.py     scaled dot-product and multi-head attention
model/layers.py        feed-forward, encoder layer, decoder layer
model/transformer.py   masks and the full model
train.py               training (Task 3)
decode.py              parser, greedy decoding, beam search, readable SQL
evaluate_model.py      writes prediction files
component_accuracy.py  sel / agg / WHERE accuracy
make_samples.py        samples.md and failure counts
plot_attention.py      cross-attention map
attention_stats.py     attention statistics over many examples
gold_roundtrip.py      parser check on the gold targets
plot_lr.py, plot_loss.py   figures
test_*.py              unit tests for every component
results/               tables, figures, logs, prediction files, check outputs
model_weights/best.pt  trained checkpoint (epoch 19)
app/                   web front end (TODO)
```

## Setup

```bash
git clone https://github.com/Ehmad-7/genai_a2_text2sql.git
cd genai_a2_text2sql
python -m venv .venv && source .venv/bin/activate
pip install torch sentencepiece records babel tqdm tabulate matplotlib numpy
pip install "sqlalchemy<2"      # only if the official evaluator fails on `records`

git clone https://github.com/salesforce/WikiSQL
(cd WikiSQL && tar xjf data.tar.bz2)
ln -s ../WikiSQL starter/WikiSQL
```

## Reproduce

```bash
# data (the trained tokenizer starter/sql_sp.model is committed;
# do NOT rerun tokenizer.py, it would overwrite it)
cd starter && python data_prep.py && python check_starter.py && cd ..

# unit tests
python test_attention.py && python test_layers.py && python test_masks.py \
  && python test_model.py && python test_parser.py && python test_decode.py

# parser check on the gold targets, then the official evaluator
python gold_roundtrip.py --via_sp
(cd WikiSQL && python evaluate.py data/dev.jsonl data/dev.db ../results/dev_gold_roundtrip.jsonl)

# training (a GPU is recommended; about 64 s per epoch on a Colab T4)
python train.py --epochs 20 --ckpt_dir checkpoints
python train.py --epochs 20 --ckpt_dir checkpoints --resume     # after an interruption
python plot_lr.py
python plot_loss.py                                             # needs results/train_log.csv

# predictions (beam search is slow on a CPU; about 10 minutes for dev on a T4)
python evaluate_model.py --split dev --method greedy
python evaluate_model.py --split dev --method beam
(cd WikiSQL && python evaluate.py data/dev.jsonl data/dev.db ../results/dev_beam.jsonl)

# analysis
python component_accuracy.py results/dev_beam.jsonl
python make_samples.py
python plot_attention.py --idx 4
python attention_stats.py --n 300
```

The test split is evaluated once, with the decoding method chosen on dev:

```bash
python evaluate_model.py --split test --method beam
(cd WikiSQL && python evaluate.py data/test.jsonl data/test.db ../results/test_beam.jsonl)
```

Notes for Google Colab: select a T4 GPU, clone the repo and WikiSQL, run `data_prep.py` inside `starter/`, mount Google Drive and pass `--ckpt_dir /content/drive/MyDrive/genai_ckpt` so checkpoints survive a disconnect. The official evaluator did not run on Colab's Python 3.13 (an incompatibility in the `records` library), so the prediction files were generated on Colab and scored locally.

## Correctness checks

Causal mask, padding mask, weight sharing, attention rows, learning-rate schedule and the gold round-trip (99.49% execution accuracy) are listed in `results/correctness_checks.md`, with raw output in `results/checks/`.

## Front end

TODO: screenshot and run command (`streamlit run app/app.py`).

## Links

- Medium blog: TODO
- LinkedIn post: TODO

## References

- Vaswani et al., *Attention Is All You Need* (2017).
- Zhong, Xiong and Socher, *Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning* (2017). WikiSQL: https://github.com/salesforce/WikiSQL
