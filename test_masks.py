import torch
from model.transformer import make_pad_mask,make_causal_mask,make_tgt_mask

F,T=False,True

ids=torch.tensor([[5,6,0,0]])
pm=make_pad_mask(ids,0)
assert pm.shape==(1,1,1,4)
assert pm.tolist()==[[[[F,F,T,T]]]]

cm=make_causal_mask(4)
assert cm.shape==(1,1,4,4)
assert cm[0,0].tolist()==[
    [F,T,T,T],
    [F,F,T,T],
    [F,F,F,T],
    [F,F,F,F]
]

tm=make_tgt_mask(torch.tensor([[2,7,8,0],[2,9,3,0]]),0)
assert tm.shape==(2,1,4,4)
assert tm[0,0].tolist()==[
    [F,T,T,T],
    [F,F,T,T],
    [F,F,F,T],
    [F,F,F,T]
]

print("mask tests passed")