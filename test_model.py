import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer import Transformer

v=8000
model=Transformer(v)
model.eval()

src=torch.randint(4,v,(2,11))
tgt_in=torch.randint(4,v,(2,7))

logits=model(src,tgt_in)
assert logits.shape==(2,7,v)

assert model.out.weight is model.shared.emb.weight
assert model.encoder_input.tok is model.decoder_input.tok

n=sum(p.numel() for p in model.parameters())
print("Trainable parameters:",n)
assert n==7577600

tgt2=tgt_in.clone()
tgt2[:, -1]=(tgt2[:,-1]+1)%v
logits2=model(src,tgt2)
assert torch.allclose(logits[:,:-1],logits2[:,:-1],atol=1e-5)

src_pad=torch.cat([src,torch.zeros(2,5,dtype=torch.long)],dim=1)
logits3=model(src_pad,tgt_in)

assert torch.allclose(logits,logits3,atol=1e-5)

target=torch.randint(4,v,(2,7))
loss=F.cross_entropy(logits.reshape(-1,v),target.reshape(-1))
print("initial loss:",loss.item())
assert 8.0<loss.item()<11.0

print("Transformer test passed")