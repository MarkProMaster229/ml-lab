import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import load_dataset

class TransformerBlock(nn.Module):
    def __init__(self, sizeVector=128, numHeads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(sizeVector)
        self.attn = nn.MultiheadAttention(sizeVector, numHeads, batch_first=True)
        self.ln2 = nn.LayerNorm(sizeVector)
        self.ff = nn.Sequential(
            nn.Linear(sizeVector, sizeVector * 4),
            nn.GELU(),
            nn.Linear(sizeVector * 4, sizeVector),
        )

    def forward(self, x, attMask=None):
        h = self.ln1(x)
        z, _ = self.attn(h, h, h, attn_mask=attMask)
        x = x + z

        h = self.ln2(x)
        z = self.ff(h)
        x = x + z

        return x
    


class TransformerRun(nn.Module):
    def __init__(self, vocabSize=120000, maxLong=256, sizeVector=128, block=4):
        super().__init__()
        self.maxLong = maxLong
        self.tokenEmbed = nn.Embedding(vocabSize, sizeVector)
        self.posEmbed = nn.Embedding(maxLong, sizeVector)
        self.ln_f = nn.LayerNorm(sizeVector)

        self.layers = nn.ModuleList([
            TransformerBlock(sizeVector=sizeVector, numHeads=4)
            for _ in range(block)
        ])

        self.lmHead = nn.Linear(sizeVector, vocabSize)
        mask = torch.triu(torch.full((maxLong, maxLong), float('-inf')), diagonal=1)
        self.register_buffer("attMask", mask)

    def forward(self, x):
        B, T = x.shape
        tok = self.tokenEmbed(x)
        pos = self.posEmbed(torch.arange(T, device=x.device)).unsqueeze(0)
        h = tok + pos

        #Правильная causal mask
        # causal mask была сломана !!!!!!!
        #causal maskcausal maskcausal maskcausal maskcausal maskcausal maskcausal mask
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        attMask = self.attMask[:T, :T]
        for layer in self.layers:
            h = layer(h, attMask)

        h = self.ln_f(h)
        return self.lmHead(h)
