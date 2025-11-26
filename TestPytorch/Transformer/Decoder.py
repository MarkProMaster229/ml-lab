import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class Transformer(nn.Module):
    
    def __init__(self):
        super(Transformer, self).__init__()
    
    
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
        