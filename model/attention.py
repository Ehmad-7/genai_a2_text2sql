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
    

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,h):
        super().__init__()
        assert d_model % h==0
        
        self.d_model=d_model
        self.h=h
        self.d_k=d_model//h
        
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        
        self.w_o=nn.Linear(d_model,d_model)
        
        self.attention_weights=None
        
    def forward(self,query,key,value,mask=None):
        batch_size=query.shape[0]
        
        q=self.w_q(query)
        k=self.w_k(key)
        v=self.w_v(value)
        
        q=q.view(batch_size,-1,self.h,self.d_k).transpose(1,2)
        k=k.view(batch_size,-1,self.h,self.d_k).transpose(1,2)
        v=v.view(batch_size,-1,self.h,self.d_k).transpose(1,2)
        
        output,weights=scaled_dot_product_attention(q,k,v,mask)
        self.attention_weights=weights.detach()
        
        output=output.transpose(1,2).contiguous()
        output=output.view(batch_size,-1,self.d_model)
        
        output=self.w_o(output)
        return output


          
    