# evaluate_model.py  (project root)
# Writes one JSON line per example, same order as the split file:
#   {"query": {...}}  or  {"error": "parse"}
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "starter"))

import torch
import sentencepiece as spm
from dataset import make_loader
from tokenizer import PAD_ID
from decode import load_model, greedy_decode, beam_search, ids_to_text, parse_query

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="dev", choices=["dev", "test"])
ap.add_argument("--method", default="greedy", choices=["greedy", "beam"])
ap.add_argument("--beam_size", type=int, default=4)
ap.add_argument("--ckpt", default="model_weights/best.pt")
ap.add_argument("--batch_size", type=int, default=64)
ap.add_argument("--limit", type=int, default=0, help="debug only: stop after N examples")
ap.add_argument("--out", default=None)
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sp = spm.SentencePieceProcessor(model_file="starter/sql_sp.model")
model = load_model(args.ckpt, sp.get_piece_size(), device)

loader = make_loader(f"starter/{args.split}_pairs.jsonl", sp, train=False,
                     batch_size=args.batch_size)          # keeps every row, in file order
os.makedirs("results", exist_ok=True)
out_path = args.out or f"results/{args.split}_{args.method}.jsonl"

n, failed, t0 = 0, 0, time.time()
with open(out_path, "w", encoding="utf-8") as f:
    for src, _ in loader:
        src = src.to(device)
        if args.method == "greedy":
            seqs = greedy_decode(model, src)
        else:
            seqs = [beam_search(model, row[row != PAD_ID].unsqueeze(0), args.beam_size)
                    for row in src]
        for ids in seqs:
            q = parse_query(ids_to_text(sp, ids))
            if q is None:
                f.write(json.dumps({"error": "parse"}) + "\n")
                failed += 1
            else:
                f.write(json.dumps({"query": q}) + "\n")
            n += 1
        print(f"\r{n} examples, {time.time() - t0:.0f}s", end="", flush=True)
        if args.limit and n >= args.limit:
            break

print(f"\nlines written: {n} | parse failures: {failed} ({100 * failed / n:.2f}%)")
print("wrote", out_path)