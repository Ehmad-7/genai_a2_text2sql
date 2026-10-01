import os
import sys
sys.path.insert(0,"starter")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sentencepiece as spm
from tokenizer import read_pairs, BOS_ID,EOS_ID
from embeddings import PositionalEncoding

RESULTS="results"
os.makedirs(RESULTS,exist_ok=True)

MAX_SRC, MAX_TGT=160,64
SPLITS=["train","dev","test"]

def load_sp():
    sp=spm.SentencePieceProcessor(model_file="starter/sql_sp.model")
    return sp

def split_stats(split,sp):
    pairs=read_pairs(f"starter/{split}_pairs.jsonl")
    src_lens,tgt_lens=[],[]
    dropped=0
    for p in pairs:
        s_len=len(sp.encode(p["src"]))+1
        t_len=len(sp.encode(p["tgt"]))+2
        if s_len>MAX_SRC or t_len>MAX_TGT:
            dropped+=1
        
        src_lens.append(s_len)
        tgt_lens.append(t_len)
    return {
        "pairs":(len(pairs)),
        "src_max":(max(src_lens)),
        "src_mean":(np.mean(src_lens)),
        "tgt_max":(max(tgt_lens)),
        "tgt_mean":(np.mean(tgt_lens)),
        "dropped":(dropped)
    }
    
    
def write_table1(stats):
    def row(label,fn):
        return f"| {label} | " + " | ".join(fn(stats[split]) for split in SPLITS) + " |"
    
    lines=[
        "| | Train | Dev | Test |",
        "|----|-------|-----|------|",
        row("Pairs",lambda s: str(s["pairs"])),
        row("Mean / Max source length(tokens)",
            lambda s:f"{s['src_mean']:.1f} / {s['src_max']}"),
        row("Mean / max target length(tokens)",
            lambda s:f"{s['tgt_mean']:.1f} / {s['tgt_max']}"),
        
        f"| Pairs dropped as too long | {stats['train']['dropped']} | - | - |",
        "",
        "Lengths include </s> on the source and <s> and </s> on the target, "
        "and are computed over all pairs before dropping "
        f"(train pairs are dropped if source > {MAX_SRC} or target > {MAX_TGT}).",
    ]
    
    text="\n".join(lines)
    print(text)
    with open(os.path.join(RESULTS,"table1_data.md"),"w",encoding="utf-8") as f:
        f.write(text + "\n")
        
        
        
def plot_positional_encoding():
    pe=PositionalEncoding(256).pe[0,:100,:].numpy()
    print("Positional Encoding shape",pe.shape)
    
    plt.figure(figsize=(10,5))
    plt.imshow(pe,aspect="auto",cmap="coolwarm")
    plt.colorbar(label="value")
    plt.xlabel("Embedding dimension (0-255)")
    plt.ylabel("Position (0-99)")
    plt.title("Sinusoidal positional encoding")
    plt.savefig(os.path.join(RESULTS,"fig1_positional_encoding.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    
if __name__=="__main__":
    
    sp=load_sp()
    stats={s:split_stats(s,sp) for s in SPLITS}
    write_table1(stats)
    plot_positional_encoding()
            
        