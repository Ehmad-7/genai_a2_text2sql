import torch
import torch.nn as nn
from model.attention import MultiHeadAttention

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff:int,dropout:float):
        super().__init__()
        self.linear_1=nn.Linear(d_model,d_ff)
        
        self.activation=nn.ReLU()
        
        self.dropout=nn.Dropout(dropout)
        
        self.linear_2=nn.Linear(d_ff,d_model)
        
    def forward(self,x:torch.Tensor):
        x=self.linear_1(x)
        x=self.activation(x)
        x=self.dropout(x)
        x=self.linear_2(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self,d_model,h,d_ff,dropout):
        super().__init__()
        self.self_attn=MultiHeadAttention(d_model,h)
        self.ffn=PositionwiseFeedForward(d_model,d_ff,dropout)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x,src_mask):
        x=self.norm1(x+self.dropout(self.self_attn(x,x,x,src_mask)))
        x=self.norm2(x+self.dropout(self.ffn(x)))
        return x
    
class Encoder(nn.Module):
    def __init__(self,N,d_model,h,d_ff,dropout):
        super().__init__()
        self.layers=nn.ModuleList(
            [EncoderLayer(d_model,h,d_ff,dropout) for _ in range(N)]
        )
        
    def forward(self,x,src_mask):
        for layer in self.layers:
            x=layer(x,src_mask)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self,d_model,h,d_ff,dropout):
        super().__init__()
        self.self_attn=MultiHeadAttention(d_model,h)
        self.cross_attn=MultiHeadAttention(d_model,h)
        self.ffn=PositionwiseFeedForward(d_model,d_ff,dropout)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x,memory,src_mask,tgt_mask):
        x=self.norm1(x+self.dropout(self.self_attn(x,x,x,tgt_mask)))
        x=self.norm2(x+self.dropout(self.cross_attn(x,memory,memory,src_mask)))
        x=self.norm3(x+self.dropout(self.ffn(x)))
        return x
    
    
class Decoder(nn.Module):
    def __init__(self,N,d_model,h,d_ff,dropout):
        super().__init__()
        self.layers=nn.ModuleList(
            [DecoderLayer(d_model,h,d_ff,dropout) for _ in range(N)]
        )
        
    def forward(self,x,memory,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,memory,src_mask,tgt_mask)
        return x
            