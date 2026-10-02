import torch
from model.layers import PositionwiseFeedForward,EncoderLayer,Encoder

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