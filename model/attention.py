import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(q,k,v,mask=None):
    d_k=q.shape[-1]
    attention_scores=torch.matmul(q,k.transpose(-2,-1))
    attention_scores=attention_scores/math.sqrt(d_k)
    
    if mask is not None:
        attention_scores=attention_scores.masked_fill(mask==True,-1e9)
    
    attention_weights=torch.softmax(attention_scores,dim=-1)
    
    output=torch.matmul(attention_weights,v)
    
    return output,attention_weights
    