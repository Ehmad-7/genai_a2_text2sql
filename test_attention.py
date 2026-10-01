import torch
from model.attention import scaled_dot_product_attention

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