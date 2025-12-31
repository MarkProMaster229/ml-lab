import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class TransformerBlock(nn.Module):
    def __init__(self, sizeVector = 128, numHeads = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(sizeVector)
        self.attn = nn.MultiheadAttention(sizeVector, numHeads, batch_first=True)
        self.ln2 = nn.LayerNorm(sizeVector)
        self.ff = nn.Sequential(
            nn.Linear(sizeVector, sizeVector*4),
            nn.GELU(),
            nn.Linear(sizeVector*4, sizeVector)
        )

    
    def forward(self, x, key_padding_mask=None):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + attn_out

        h = self.ln2(x)
        x = x + self.ff(h)

        return x
    
class TransformerRun(nn.Module):
    def __init__(self, vocabSize = 120000, maxLong = 100, sizeVector = 128,block = 4):
        super().__init__()
        self.maxLong = maxLong 
        self.Vectorization = nn.Embedding(vocabSize, sizeVector)
        self.posEmbed = nn.Embedding(maxLong, sizeVector)
        self.layers = nn.ModuleList([
            TransformerBlock(sizeVector=sizeVector, numHeads=8)
            for _ in range(block)
            ])
        
        self.lmHead = nn.Linear(sizeVector, 3)#да три выхода 

    def forward(self, x):
        B, T = x.shape
        tok = self.Vectorization(x)
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pos = self.posEmbed(positions)


        h = tok + pos# нужна позиционка 

        cls = h.mean(dim=1)
        
        logits = self.lmHead(cls)
        return logits