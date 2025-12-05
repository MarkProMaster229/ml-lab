import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
class TransformerBlock(nn.Module):
    def __init__(self, sizeVector = 512, numHeads = 16):
        super().__init__()
        #первый слой
        #слой внимания 
        #нормализация 
        #второй слой
        #слой внимания
        #нормализация 
        #третий слой  
        self.sizeVector = sizeVector
        self.ln1 = nn.LayerNorm(sizeVector)
        self.attn = nn.MultiheadAttention(sizeVector, numHeads, batch_first=True)
        self.ln2 = nn.LayerNorm(sizeVector)
        self.attn2 = nn.MultiheadAttention(sizeVector, numHeads, batch_first=True)
        self.ln3 = nn.LayerNorm(sizeVector)
        self.ff = nn.Sequential(
            nn.Linear(sizeVector, sizeVector*4),
            nn.GELU(),
            nn.Linear(sizeVector*4, sizeVector)
        )
    def forward(self,x, attMask = None):
        h1 = self.ln1(x)
        z1 = self.attn(h1,h1,h1, attn_mask = attMask)[0]
        x = x + z1

        h2 = self.ln2(x)
        z2 = self.attn2(h2,h1,h1, attn_mask = attMask)[0]#очень интересно, внимание на внимание
        x = x + z2 
        
        #FeedForward
        ret = self.ln3(x)
        z3 = self.ff(ret)
        x = x + z3 
        return x

class TransformerRun(nn.Module):
    def __init__(self, vocabSize = 120000, maxLong = 100, sizeVector = 512,block = 15):
        super().__init__()
        self.maxLong = maxLong 
        self.Vectorization = nn.Embedding(vocabSize, sizeVector)
        self.posEmbed = nn.Embedding(maxLong, sizeVector)
        self.layers = nn.ModuleList([
            TransformerBlock(sizeVector=sizeVector, numHeads=16)
            for _ in range(block)
            ])
        self.lmHead = nn.Linear(sizeVector,vocabSize)

    def forward(self, x):
        B, T = x.shape
        tok = self.Vectorization(x)
        pos = self.posEmbed(torch.arange(T,device=x.device))
        pos = pos.unsqueeze(0).repeat(B, 1, 1)

        h = tok + pos

        attMask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        for layer in self.layers:
            h = layer(h, attMask=attMask)


        return self.lmHead(h)