import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class TransformerBlock(nn.Module):
    def __init__(self, sizeVector = 64, numHeads = 2):
        super().__init__()
        self.sizeVector = sizeVector
        self.ln1 = nn.LayerNorm(sizeVector)
        self.attn = nn.MultiheadAttention(sizeVector, numHeads, batch_first=True)
        self.ln2 = nn.LayerNorm(sizeVector)
        self.ff = nn.Sequential(
            nn.Linear(sizeVector, sizeVector*4),
            nn.GELU(),
            nn.Linear(sizeVector*4, sizeVector),
        )

    def forward(self, x, attMask = None):
        h = self.ln1(x)
        z, _ = self.attn(h, h, h, attn_mask=attMask)

        h = self.ln2(x)
        z1 = self.ff(h)
        x = x + z1
        return x 
    
class TransformerRun(nn.Module):
    def __init__(self, vocabSize = 120000, maxLong = 100, sizeVector = 64 ,block = 2):
        super().__init__()
        self.maxLong = maxLong
        self.tokenEmbed = nn.Embedding(vocabSize, sizeVector)
        self.posEmbed   = nn.Embedding(maxLong, sizeVector)
        self.ln_f = nn.LayerNorm(sizeVector)


        self.layers = nn.ModuleList([
            TransformerBlock(sizeVector=sizeVector, numHeads=2)
            for _ in range(block)
            ])

        self.lmHead = nn.Linear(sizeVector,vocabSize)
    def forward(self, x):
        B,T = x.shape
        tok = self.tokenEmbed(x)
        pos = self.posEmbed(torch.arange(T,device=x.device))
        pos = pos.unsqueeze(0).repeat(B, 1, 1)

        h = tok + pos

        attMask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        for layer in self.layers:
            h = layer(h, attMask=attMask)
        h = self.ln_f(h)

        return self.lmHead(h)