import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "starter"))

import torch
from tokenizer import PAD_ID, BOS_ID, EOS_ID
from data_prep import AGG_OPS, COND_OPS
from model.transformer import Transformer, make_pad_mask, make_tgt_mask

AGG_WORDS=[a.lower() for a in AGG_OPS]

def parse_query(text):
    text=text.strip()
    select_pattern = r"^select\s+(?:(\w+)\s+)?<c(\d+)>(.*)$"
    select_match = re.match(select_pattern, text, re.IGNORECASE)
    if not select_match:
        return None
    
    agg_word,sel_col,rest=select_match.groups()
    sel=int(sel_col)
    agg=0
    
    if agg_word:
        agg_word_lower=agg_word.lower()
        if agg_word_lower in AGG_WORDS:
            agg=AGG_WORDS.index(agg_word_lower)
        else:
            return None
        
    rest=rest.strip()
    
    if not rest:
        return {"sel":sel,"agg":agg,"conds":[]}
    
    if not rest.lower().startswith("where"):
        return None
    
    cond_pattern = r"\b(where|and)\s+<c(\d+)>\s+([=><])"
    matches = list(re.finditer(cond_pattern, rest, re.IGNORECASE))
    if not matches or matches[0].start() != 0 or matches[0].group(1).lower() != "where":
        return None        
    conds=[]
    for i,match in enumerate(matches):
        col=int(match.group(2))
        op=match.group(3)
        
        start_val=match.end()
        end_val=matches[i+1].start() if i+1<len(matches) else len(rest)
        
        value=rest[start_val:end_val].strip()
        
        if not value:
            return None
        
        conds.append([col,COND_OPS.index(op),value])
        
    return {"sel":sel,"agg":agg,"conds":conds}

def load_model(ckpt_path,vocab_size,device):
    model=Transformer(vocab_size)
    ckpt=torch.load(ckpt_path,map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()

def cut_at_eos(ids):
    return ids[:ids.index(EOS_ID)+1] if EOS_ID in ids else ids

def ids_to_text(sp,ids):
    if EOS_ID in ids:
        ids=ids[:ids.index(EOS_ID)]
    return sp.decode([i for i in ids if i!=BOS_ID])

@torch.no_grad()
def greedy_decode(model,src,max_len=64):
    B,device= src.size(0),src.device
    src_mask=make_pad_mask(src,PAD_ID)
    memory=model.encode(src,src_mask)
    ys=torch.full((B,1),BOS_ID,dtype=torch.long,device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for i in range(max_len-1):
        tgt_mask=make_tgt_mask(ys,PAD_ID)
        logits=model.decode(ys,memory,src_mask,tgt_mask)
        next_tok=logits[:,-1, :].argmax(dim=-1)
        next_tok=torch.where(finished,torch.tensor(PAD_ID,device=device),next_tok)
        ys=torch.cat([ys,next_tok.unsqueeze(1)],dim=1)
        finished=finished | (next_tok==EOS_ID)
        if finished.all():
            break
    return [cut_at_eos(row[1:].tolist()) for row in ys]

@torch.no_grad()
def beam_search(model,src,beam_size=4,max_len=64):
    
    device=src.device
    
    src_mask=make_pad_mask(src,PAD_ID)
    memory_single=model.encode(src,src_mask)
    
    memory=memory_single.expand(beam_size,-1,-1)
    src_mask=src_mask.expand(beam_size,-1,-1,-1)
    
    ys=torch.full((beam_size,1),BOS_ID,dtype=torch.long,device=device)
    scores=torch.zeros(beam_size,dtype=torch.float,device=device)
    scores[1:]=-1e9
    
    finished=[]
    
    for _ in range(max_len):
        tgt_mask=make_tgt_mask(ys,PAD_ID)
        
        logits=model.decode(ys,memory,src_mask,tgt_mask)
        log_probs=torch.log_softmax(logits[:, -1, :],dim=-1)
        
        total_scores=scores.unsqueeze(1)+log_probs
        
        vocab_size=total_scores.size(-1)
        topk_scores,topk_indices=total_scores.view(-1).topk(beam_size)
        
        beam_idx=torch.div(topk_indices,vocab_size,rounding_mode="floor")
        token=topk_indices%vocab_size
        
        ys=torch.cat([ys[beam_idx],token.unsqueeze(1)],dim=1)
        scores=topk_scores
        
        alive_mask=token!=EOS_ID
        
        for i in range(beam_size):
            if not alive_mask[i]:
                hyp_ids=ys[i,1:].tolist()
                finished.append((scores[i].item(),hyp_ids))
                
        scores=torch.where(alive_mask,scores,torch.tensor(-1e9,device=device))
        
        if len(finished)>=beam_size:
            break
        
    if finished:
        finished.sort(key=lambda x: x[0], reverse=True)
        return finished[0][1]
    return ys[scores.argmax(), 1:].tolist()     
        