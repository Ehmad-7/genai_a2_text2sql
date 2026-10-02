import torch
from model.layers import PositionwiseFeedForward,EncoderLayer,Encoder,DecoderLayer,Decoder
from model.transformer import make_causal_mask,make_pad_mask

ffn=PositionwiseFeedForward(256,1024,0.1)
x=torch.randn(2,7,256)
assert ffn(x).shape==(2,7,256)

n=sum(p.numel() for p in ffn.parameters())
assert n==525568,n

print("ffn tests passed")

layer=EncoderLayer(256,4,1024,0.1)
x=torch.randn(2,7,256)
mask=torch.zeros(2,1,1,7,dtype=torch.bool)
assert layer(x,mask).shape==(2,7,256)
n=sum(p.numel() for p in layer.parameters())

assert n==789760,n

enc=Encoder(3,256,4,1024,0.1)
assert enc(x,mask).shape==(2,7,256)
n=sum(p.numel() for p in enc.parameters())
assert n==2369280,n

assert enc.layers[0].self_attn.w_q.weight is not enc.layers[1].self_attn.w_q.weight

enc.eval()
mask=torch.zeros(2,1,1,7,dtype=torch.bool)
mask[...,-2:]=True
out1=enc(x,mask)
x2=x.clone()
x2[:,-2:,:]=torch.randn(2,2,256)
out2=enc(x2,mask)

assert torch.allclose(out1[:,:-2],out2[:,:-2],atol=1e-5)

print("encoder tests passed")

## Decoder tests

d1=DecoderLayer(256,4,1024,0.1)
x=torch.randn(2,6,256)
memory=torch.randn(2,9,256)
src_mask=torch.zeros(2,1,1,9,dtype=torch.bool)
tgt_mask=make_causal_mask(6)
assert d1(x,memory,src_mask,tgt_mask).shape==(2,6,256)
assert sum(p.numel() for p in d1.parameters())==1053440

dec=Decoder(3,256,4,1024,0.1)
assert dec(x,memory,src_mask,tgt_mask).shape==(2,6,256)
assert sum(p.numel() for p in dec.parameters())==3160320

dec.eval()
out1=dec(x,memory,src_mask,tgt_mask)
x2=x.clone()
x2[:,-1,:]=torch.randn(2,256)
out2=dec(x2,memory,src_mask,tgt_mask)
assert torch.allclose(out1[:,:-1],out2[:,:-1],atol=1e-5)
assert not torch.allclose(out1[:,-1],out2[:,-1],atol=1e-5)

src_mask2=torch.zeros(2,1,1,9,dtype=torch.bool)
src_mask2[...,-2:]=True
out1=dec(x,memory,src_mask2,tgt_mask)
memory2=memory.clone()
memory2[:,-2:, :]=torch.randn(2,2,256)
out2=dec(x,memory2,src_mask2,tgt_mask)
assert torch.allclose(out1,out2,atol=1e-5)

print("decoder tests passed")