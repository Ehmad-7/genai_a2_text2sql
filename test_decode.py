import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "starter"))

import torch
import sentencepiece as spm
from tokenizer import read_pairs, PAD_ID, EOS_ID
from decode import load_model, greedy_decode, beam_search, ids_to_text

sp = spm.SentencePieceProcessor(model_file="starter/sql_sp.model")
model = load_model("model_weights/best.pt", sp.get_piece_size(), "cpu")

pairs = read_pairs("starter/dev_pairs.jsonl")[:8]
rows = [sp.encode(p["src"]) + [EOS_ID] for p in pairs]
S = max(map(len, rows))
src = torch.tensor([r + [PAD_ID] * (S - len(r)) for r in rows])

batch_out = greedy_decode(model, src)
for p, ids in zip(pairs[:5], batch_out):
    print("gold  :", p["tgt"])
    print("greedy:", ids_to_text(sp, ids), "\n")

# 2. padding invariance: batched result equals decoding each row alone, unpadded
for r, ids in zip(rows, batch_out):
    alone = greedy_decode(model, torch.tensor([r]))[0]
    assert alone == ids, "batched greedy differs from single-example greedy"

# 3. beam search with beam_size=1 must equal greedy
for r, ids in zip(rows, batch_out):
    b1 = beam_search(model, torch.tensor([r]), beam_size=1)
    assert b1 == ids, "beam_size=1 differs from greedy"

# 4. beam 4 returns something sensible and respects the length limit
for r in rows:
    b4 = beam_search(model, torch.tensor([r]), beam_size=4)
    assert 0 < len(b4) <= 65

print("decode tests passed")