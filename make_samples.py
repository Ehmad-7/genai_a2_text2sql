import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "starter"))
from decode import to_sql

PRED = sys.argv[1] if len(sys.argv) > 1 else "results/dev_beam.jsonl"


def norm(v):
    return str(v).strip().lower()


def cond_set(conds):
    return {(c, o, norm(v)) for c, o, v in conds}


def failure(gold, q):
    """None if the logical form is correct, otherwise the name of the failure.
    Checked in this order, so each wrong example gets exactly one label."""
    if q is None:
        return "parse failure"
    if q["sel"] != gold["sel"]:
        return "wrong select column"
    if q["agg"] != gold["agg"]:
        return "wrong aggregation"
    g, p = cond_set(gold["conds"]), cond_set(q["conds"])
    if g == p:
        return None
    if len(p) < len(g):
        return "missing condition"
    if len(p) > len(g):
        return "extra condition"
    if {c for c, _, _ in g} != {c for c, _, _ in p}:
        return "wrong column in WHERE"
    if {(c, o) for c, o, _ in g} != {(c, o) for c, o, _ in p}:
        return "wrong operator"
    return "wrong value"


tables = {}
with open("WikiSQL/data/dev.tables.jsonl", encoding="utf-8") as f:
    for line in f:
        t = json.loads(line)
        tables[t["id"]] = t

recs = []
with open("WikiSQL/data/dev.jsonl", encoding="utf-8") as fg, \
     open(PRED, encoding="utf-8") as fp:
    for i, (gl, pl) in enumerate(zip(fg, fp)):
        g, p = json.loads(gl), json.loads(pl)
        header = tables[g["table_id"]]["header"]
        q = p.get("query")
        recs.append({
            "i": i,
            "question": g["question"],
            "header": header,
            "gold_sql": to_sql(g["sql"], header),
            "pred_sql": to_sql(q, header) if q else "(could not parse the model output)",
            "fail": failure(g["sql"], q),
        })

assert len(recs) == 8421, len(recs)

counts = Counter(r["fail"] for r in recs if r["fail"])
n_correct = sum(r["fail"] is None for r in recs)
print("failure types over all of dev:", dict(counts))
print("logical-form correct:", n_correct, "of", len(recs))

rng = random.Random(0)                       # fixed seed: the same examples every run
correct = [r for r in recs if r["fail"] is None]
wrong = [r for r in recs if r["fail"]]
rng.shuffle(wrong)

picked_wrong, seen = [], set()
for r in wrong:                              # first, one example per failure type
    if r["fail"] not in seen and len(picked_wrong) < 5:
        picked_wrong.append(r)
        seen.add(r["fail"])
for r in wrong:                              # then fill up to 5
    if len(picked_wrong) < 5 and r not in picked_wrong:
        picked_wrong.append(r)

lines = [
    "# Qualitative samples (dev, beam search k=4)",
    "",
    "Five correct and five wrong examples, chosen with a fixed random seed. "
    "SQL uses the real column names. A wrong example is counted under its first "
    "failing component (parse, select column, aggregation, WHERE).",
    "",
]
for title, group in [("Correct", rng.sample(correct, 5)), ("Wrong", picked_wrong)]:
    lines += [f"## {title}", ""]
    for r in group:
        lines += [
            f"**Dev example {r['i']}**",
            "",
            f"- Question: {r['question']}",
            f"- Columns: {', '.join(r['header'])}",
            f"- Gold SQL: `{r['gold_sql']}`",
            f"- Our SQL: `{r['pred_sql']}`",
        ]
        if r["fail"]:
            lines.append(f"- Failure: **{r['fail']}**")
        lines.append("")

Path("results").mkdir(exist_ok=True)
Path("results/samples.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote results/samples.md")