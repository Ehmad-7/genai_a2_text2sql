import torch

def make_pad_mask(ids,pad_id):
    return (ids==pad_id).unsqueeze(1).unsqueeze(2)

def make_causal_mask(T,device=None):
    mask=torch.ones((T,T),dtype=torch.bool,device=device)
    return torch.triu(mask,diagonal=1).unsqueeze(0).unsqueeze(0)

def make_tgt_mask(tgt_ids,pad_id):
    pad_mask=make_pad_mask(tgt_ids,pad_id)
    causal_mask=make_causal_mask(tgt_ids.size(1),device=tgt_ids.device)
    return pad_mask | causal_mask
    