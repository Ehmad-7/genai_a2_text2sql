import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "starter"))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sentencepiece as spm
from tokenizer import read_pairs, BOS_ID, EOS_ID
from decode import load_model, beam_search, ids_to_text

ap = argparse.ArgumentParser()
ap.add_argument("--idx", type=int, default=0, help="dev example index")
args = ap.parse_args()

sp = spm.SentencePieceProcessor(model_file="starter/sql_sp.model")
model = load_model("model_weights/best.pt", sp.get_piece_size(), "cpu")

pair = read_pairs("starter/dev_pairs.jsonl")[args.idx]
src_ids = sp.encode(pair["src"]) + [EOS_ID]
src = torch.tensor([src_ids])

gen = beam_search(model, src, beam_size=4)           # generated ids, ends with </s>
tgt_in = torch.tensor([[BOS_ID] + gen[:-1]])         # feed the model its own output
with torch.no_grad():
    model(src, tgt_in)                               # clean pass; weights are stored on the module

w = model.decoder.layers[-1].cross_attn.attention_weights   # (1, heads, T, S)
attn = w[0].mean(dim=0).numpy()                      # average over heads -> (T, S)

src_lab = [sp.id_to_piece(i) for i in src_ids]
tgt_lab = [sp.id_to_piece(i) for i in gen]

print("gold     :", pair["tgt"])
print("generated:", ids_to_text(sp, gen))

# Which source tokens belong to which column? <cK> token up to the next column token.
cols = [(i, t) for i, t in enumerate(src_lab) if t.startswith("<c") and t.endswith(">")]
spans = {t: (i, cols[k + 1][0] if k + 1 < len(cols) else len(src_lab))
         for k, (i, t) in enumerate(cols)}

print("\nWhere does each generated <cK> token look?")
for r, t in enumerate(tgt_lab):
    if t in spans:
        j = int(attn[r].argmax())
        a, b = spans[t]
        print(f"  {t}: peak on source token {j} ({src_lab[j]!r}), "
              f"inside its own column span: {a <= j < b}")

fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(src_lab)), max(3, 0.35 * len(tgt_lab))))
im = ax.imshow(attn, aspect="auto", cmap="viridis")
ax.set_xticks(range(len(src_lab)))
ax.set_xticklabels(src_lab, rotation=90, fontsize=7)
ax.set_yticks(range(len(tgt_lab)))
ax.set_yticklabels(tgt_lab, fontsize=8)
ax.set_xlabel("source tokens")
ax.set_ylabel("generated tokens")
ax.set_title(f"Decoder cross-attention, last layer, mean over heads (dev example {args.idx})")
fig.colorbar(im, ax=ax)
fig.savefig("results/fig4_attention_map.png", dpi=150, bbox_inches="tight")
print("\nwrote results/fig4_attention_map.png")