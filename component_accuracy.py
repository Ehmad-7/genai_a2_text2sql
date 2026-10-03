import json, sys

PRED = sys.argv[1] if len(sys.argv) > 1 else "results/dev_beam.jsonl"
GOLD = "WikiSQL/data/dev.jsonl"

def norm(v):
    return str(v).strip().lower()

n = sel_ok = agg_ok = where_ok = 0
with open(GOLD, encoding="utf-8") as fg, open(PRED, encoding="utf-8") as fp:
    for g_line, p_line in zip(fg, fp):               
        gold = json.loads(g_line)["sql"]             
        pred = json.loads(p_line)                    
        n += 1
        if "query" not in pred:                     
            continue
        q = pred["query"]
        
        
        if q.get("sel") == gold.get("sel"):
            sel_ok += 1
            
        
        if q.get("agg") == gold.get("agg"):
            agg_ok += 1
            
        
        gold_conds = set((col, op, norm(val)) for col, op, val in gold.get("conds", []))
        pred_conds = set((col, op, norm(val)) for col, op, val in q.get("conds", []))
        
        if gold_conds == pred_conds:
            where_ok += 1
assert n==8421

print(f"examples: {n}")
print(f"sel   correct: {100 * sel_ok / n:.2f}%")
print(f"agg   correct: {100 * agg_ok / n:.2f}%")
print(f"where correct: {100 * where_ok / n:.2f}%")
