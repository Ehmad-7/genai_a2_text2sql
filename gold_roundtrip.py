import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "starter"))

import sentencepiece as spm
from decode import parse_query

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="dev")
ap.add_argument("--via_sp", action="store_true",
                help="encode and decode each target with SentencePiece first "
                     "(this is what model output goes through)")
ap.add_argument("--out", default=None)
args = ap.parse_args()

pairs_path = f"starter/{args.split}_pairs.jsonl"
out_path = args.out or f"results/{args.split}_gold_roundtrip.jsonl"
os.makedirs("results", exist_ok=True)

sp = spm.SentencePieceProcessor(model_file="starter/sql_sp.model") if args.via_sp else None

n, failed = 0, 0
failures = []
with open(pairs_path, encoding="utf-8") as f_in, \
     open(out_path, "w", encoding="utf-8") as f_out:
    for line in f_in:                      
        tgt = json.loads(line)["tgt"]
        if sp is not None:
            tgt = sp.decode(sp.encode(tgt))
        q = parse_query(tgt)
        if q is None:
            f_out.write(json.dumps({"error": "parse"}) + "\n")
            failed += 1
            if len(failures) < 5:
                failures.append(tgt)
        else:
            f_out.write(json.dumps({"query": q}) + "\n")
        n += 1

print(f"lines written: {n}  (dev should have 8421)")
print(f"parse failures: {failed} ({100 * failed / n:.2f}%)")
for t in failures:
    print("  failed on:", t)
print("wrote", out_path)