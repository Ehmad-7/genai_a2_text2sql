import torch
from model.attention import scaled_dot_product_attention
from model.attention import MultiHeadAttention

B,h,L,d_k=2,4,5,64

q=torch.randn(B,h,L,d_k)
k=torch.randn(B,h,L,d_k)
v=torch.randn(B,h,L,d_k)

out,w=scaled_dot_product_attention(q,k,v)
print(out.shape,w.shape)

assert out.shape==(B,h,L,d_k)
assert w.shape==(B,h,L,L)

assert torch.allclose(w.sum(-1),torch.ones(B,h,L),atol=1e-5)
pad_mask=torch.zeros(1,1,1,L,dtype=torch.bool)

pad_mask[...,-2:]=True
out,w=scaled_dot_product_attention(q,k,v,pad_mask)

assert w[...,-2:].abs().max()<1e-6
assert torch.allclose(w.sum(-1),torch.ones(B,h,L),atol=1e-5)

casual=torch.triu(torch.ones(L,L),diagonal=1).bool()
out,w=scaled_dot_product_attention(q,k,v,casual)
assert torch.triu(w,diagonal=1).abs().max()<1e-6
assert torch.allclose(w.sum(-1),torch.ones(B,h,L),atol=1e-5)

q2=torch.randn(B,h,3,d_k)
out,w=scaled_dot_product_attention(q2,k,v)
assert out.shape==(B,h,3,d_k)
assert w.shape==(B,h,3,L)

print("All attention tests passed")

mha=MultiHeadAttention(256,4)

x=torch.randn(2,7,256)
mha_output=mha(x,x,x)

assert mha_output.shape==(2,7,256)
assert mha.attention_weights.shape==(2,4,7,7)

qx=torch.randn(2,5,256)
kv=torch.randn(2,7,256)
mhax_out=mha(qx,kv,kv)

assert mhax_out.shape==(2,5,256)
assert mha.attention_weights.shape==(2,4,5,7)

mask=torch.zeros(2,1,1,7,dtype=torch.bool)
mask[...,-2:]=True
mha(qx,kv,kv,mask)
assert mha.attention_weights[...,-2:].abs().max()<1e-6

n=sum(p.numel() for p in mha.parameters())
assert n==263168,n

print("multi-head attention tests passed")