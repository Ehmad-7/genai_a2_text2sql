import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

import torch
import sentencepiece as spm
from tokenizer import EOS_ID
from data_prep import encode_source, MAX_COLS
from decode import load_model, beam_search, ids_to_text, parse_query, to_sql

MAX_SRC_TOKENS = 500


@lru_cache(maxsize=1)
def load_everything():
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / "starter" / "sql_sp.model"))
    model = load_model(str(ROOT / "model_weights" / "best.pt"), sp.get_piece_size(), "cpu")
    return sp, model


def generate_sql(question: str, columns: str) -> dict:
    """Returns {"sql", "raw", "query", "error"}; never raises on bad user input."""
    sp, model = load_everything()
    header = [c.strip() for c in columns.split(",") if c.strip()]
    if not question.strip():
        return {"sql": None, "raw": "", "query": None, "error": "Please type a question."}
    if not header:
        return {"sql": None, "raw": "", "query": None, "error": "Please list the column names, separated by commas."}
    if len(header) > MAX_COLS:
        return {"sql": None, "raw": "", "query": None, "error": f"At most {MAX_COLS} columns are supported."}

    source = encode_source(question, header)
    ids = sp.encode(source) + [EOS_ID]
    if len(ids) > MAX_SRC_TOKENS:
        return {"sql": None, "raw": "", "query": None, "error": f"Input is too long (maximum {MAX_SRC_TOKENS} tokens)."}
    generated = beam_search(model, torch.tensor([ids], dtype=torch.long), beam_size=4)
    raw = ids_to_text(sp, generated)
    query = parse_query(raw)
    if query is None:
        return {"sql": None, "raw": raw, "query": None, "error": "Could not parse a valid query from the generated text."}
    return {"sql": to_sql(query, header), "raw": raw, "query": query, "error": None}