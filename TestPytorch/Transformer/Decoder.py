import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class Transformer(nn.Module):
    
    def __init__(self, vocabSize=1000, sizeVector=256, maxLong=100):
        super().__init__()
        self.sizeVector = sizeVector
        
        self.Vectorization = nn.Embedding(vocabSize, sizeVector)
        self.posEmbed = nn.Embedding(maxLong, sizeVector)
        
        self.ln1 = nn.LayerNorm(sizeVector)
        self.attn = nn.MultiheadAttention(sizeVector, 8, batch_first=True)
        self.ln2 = nn.LayerNorm(sizeVector)
        
        self.ff = nn.Sequential(
            nn.Linear(sizeVector, sizeVector*4),
            nn.GELU(),
            nn.Linear(sizeVector*4, sizeVector)
        )
    
    def inpute(self):
        batchSize = 2
        seqLen = 15
        vocabSize = 1000
        input_ids = torch.randint(0, vocabSize, (batchSize, seqLen ))
        
        #полезно благодаря нему превращаем индексы в ветора 
        #пусть размерность вектора будет 256
        sizeVector = 256
        
        Vectorization = torch.nn.Embedding(vocabSize,sizeVector)
        x = Vectorization(input_ids)
        
        maxLong = 100
        posEmbed = torch.nn.Embedding(maxLong,sizeVector)
        position = torch.arange(seqLen).unsqueeze(0)
        x = x + posEmbed(position)
        
        mask = torch.triu(torch.ones(seqLen, seqLen) * float('-inf'), diagonal=1)
        
        def forward(self, input_ids):
            