import torch
import torch.nn as nn
from model.layers import PositionwiseFeedForward,EncoderLayer,Encoder,DecoderLayer,Decoder
import sys
sys.path.insert(0,"starter")
from embeddings import TokenEmbedding,InputLayer 

def make_pad_mask(ids,pad_id):
    return (ids==pad_id).unsqueeze(1).unsqueeze(2)

def make_causal_mask(T,device=None):
    mask=torch.ones((T,T),dtype=torch.bool,device=device)
    return torch.triu(mask,diagonal=1).unsqueeze(0).unsqueeze(0)

def make_tgt_mask(tgt_ids,pad_id):
    pad_mask=make_pad_mask(tgt_ids,pad_id)
    causal_mask=make_causal_mask(tgt_ids.size(1),device=tgt_ids.device)
    return pad_mask | causal_mask



class Transformer(torch.nn.Module):
    def __init__(self,vocab_size,d_model=256,h=4,N=3,d_ff=1024,dropout=0.1,pad_id=0):
        super().__init__()
        self.pad_id=pad_id
        self.d_model=d_model
        
        self.shared=TokenEmbedding(vocab_size,d_model,pad_id)
        self.encoder_input=InputLayer(self.shared,d_model,dropout=dropout)
        self.decoder_input=InputLayer(self.shared,d_model,dropout=dropout)
        self.encoder=Encoder(N,d_model,h,d_ff,dropout)
        self.decoder=Decoder(N,d_model,h,d_ff,dropout)
        self.out=nn.Linear(d_model,vocab_size,bias=False)
        
        self.out.weight=self.shared.emb.weight
        nn.init.normal_(self.shared.emb.weight,mean=0.0,std=d_model**-0.5)
        
        for p in self.parameters():
            if p.dim()>1 and p is not self.shared.emb.weight:
                nn.init.xavier_uniform_(p)
                
        with torch.no_grad():
            self.shared.emb.weight[pad_id].zero_()
                
        
    def encode(self,src,src_mask):
        x=self.encoder_input(src)
        memory=self.encoder(x,src_mask)
        return memory
    
    def decode(self,tgt_in,memory,src_mask,tgt_mask):
        x=self.decoder_input(tgt_in)
        decoder_output=self.decoder(x,memory,src_mask,tgt_mask)
        logits=self.out(decoder_output)
        return logits
    
    def forward(self,src,tgt_in):
        src_mask=make_pad_mask(src,self.pad_id)
        tgt_mask=make_tgt_mask(tgt_in,self.pad_id)
        memory=self.encode(src,src_mask)
        return self.decode(tgt_in,memory,src_mask,tgt_mask)
    
        