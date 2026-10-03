import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "starter"))

import torch
import sentencepiece as spm
from tokenizer import read_pairs, BOS_ID, EOS_ID
from decode import load_model, greedy_decode

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
args = ap.parse_args()

sp = spm.SentencePieceProcessor(model_file="starter/sql_sp.model")
model = load_model("model_weights/best.pt", sp.get_piece_size(), "cpu")
pairs = read_pairs("starter/dev_pairs.jsonl")[:args.n]

tot = on_question = on_own_tok = in_own_span = in_any_col = 0
mass_own = mass_question = 0.0

for pair in pairs:
    src_ids = sp.encode(pair["src"]) + [EOS_ID]
    src = torch.tensor([src_ids])
    gen = greedy_decode(model, src)[0]
    tgt_in = torch.tensor([[BOS_ID] + gen[:-1]])
    with torch.no_grad():
        model(src, tgt_in)
    attn = model.decoder.layers[-1].cross_attn.attention_weights[0].mean(0).numpy()

    src_lab = [sp.id_to_piece(i) for i in src_ids]
    sep = src_lab.index("<sep>")
    cols = [(i, t) for i, t in enumerate(src_lab) if t.startswith("<c")]
    spans = {t: (i, cols[k + 1][0] if k + 1 < len(cols) else len(src_lab))
             for k, (i, t) in enumerate(cols)}

    for r, tid in enumerate(gen):
        t = sp.id_to_piece(tid)
        if t not in spans:
            continue
        a, b = spans[t]
        j = int(attn[r].argmax())
        tot += 1
        on_question += j < sep
        on_own_tok += j == a
        in_own_span += a <= j < b
        in_any_col += j > sep
        mass_own += attn[r, a:b].sum()
        mass_question += attn[r, :sep].sum()

print(f"generated <cK> tokens analysed: {tot} (from {len(pairs)} dev examples, greedy)")
print(f"peak on a question token          : {100 * on_question / tot:.1f}%")
print(f"peak anywhere in the column list  : {100 * in_any_col / tot:.1f}%")
print(f"peak on its own <cK> token        : {100 * on_own_tok / tot:.1f}%")
print(f"peak inside its own column span   : {100 * in_own_span / tot:.1f}%")
print(f"mean attention mass on own span   : {100 * mass_own / tot:.1f}%")
print(f"mean attention mass on question   : {100 * mass_question / tot:.1f}%")