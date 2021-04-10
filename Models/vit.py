# -*- coding: utf-8 -*-
"""
Please Read below Article which help me to give understanding towards Vision Transformers
https://amaarora.github.io/2021/01/18/ViT.html#cls-token--position-embeddings
"""



import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat


patch_size = 16
embed_size = 256
num_channels = 3
img_size = 224
batch_size = 2
x = torch.randn(batch_size,num_channels,img_size,img_size) 


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size = 16, embed_size = 256 ,img_size = img_size):
                
        
        super().__init__()
        
        self.proj = nn.Sequential(
                    
                    Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
                    nn.Linear(patch_size * patch_size * 3, embed_size)
            
            )
        
        self.cls_token = nn.Parameter(torch.randn(1,1, embed_size))
        
        
    def forward(self,x):
        
        b,_,_,_ = x.size()
        x = self.proj(x)
        c_tkn = repeat(self.cls_token, '() n e -> b n e', b=b)
        res =torch.cat([c_tkn,x],1)
        
        
        return res


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size = 256 ,num_heads = 2 ):       
        super().__init__()
        
        self.key = nn.Linear(embed_size, embed_size*num_heads)
        self.query= nn.Linear(embed_size, embed_size*num_heads)
        self.value= nn.Linear(embed_size, embed_size*num_heads)
        
        self.ll = nn.Linear(embed_size*num_heads ,embed_size )
        
    def forward(self,x):
        
        num_heads =2 
        
        w_key   = self.key(x)   ##[1, 197, 512]
        w_query = self.query(x) ##[1, 197, 512]
        w_value = self.value(x) ##[1, 197, 512]
        
        
        w_key     = w_key.view(batch_size,num_heads,-1,embed_size)
        w_query   = w_query.view(batch_size,num_heads,-1,embed_size)
        w_value   = w_value.view(batch_size,num_heads,-1,embed_size)
        
        energy = torch.matmul(w_key,w_query.permute(0,1,3,2))
        energy = torch.softmax(energy,1)

        f_energy = torch.einsum('bnll,bnle -> bnle',energy,w_value)
        f_energy = f_energy.permute(0,2,1,3)
        f_energy = rearrange(f_energy,' b h n e->b h (n e) ')
        
        out = self.ll(f_energy)
        
        return out
    
    

class EncorderBlock(nn.Module):
    def __init__(self,embed_size = 256 ):       
        super().__init__()
        
        self.Embedding = PatchEmbedding()
        self.attn = MultiHeadAttention()
        self.norm = nn.LayerNorm(embed_size) 
    
    def forward(self,x):
        
        x = self.Embedding(x)
        y = x
        y = self.attn(self.norm(y))
        
        res = x+y
        
        return res
    

model = EncorderBlock()

result = model(x).shape


        
        
