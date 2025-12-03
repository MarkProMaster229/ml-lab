import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
class TransformerBlock(nn.Module):
    def __init__(self, sizeVector=256, num_heads=8):
        super().__init__()
        self.sizeVector = sizeVector
        #вынеси
        #self.Vectorization = nn.Embedding(vocabSize, sizeVector)
        #вынеси
        #self.posEmbed = nn.Embedding(maxLong, sizeVector)
        self.ln1 = nn.LayerNorm(sizeVector)
        self.attn = nn.MultiheadAttention(sizeVector, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(sizeVector)

        self.ff = nn.Sequential(
            nn.Linear(sizeVector, sizeVector*4),
            nn.GELU(),
            nn.Linear(sizeVector*4, sizeVector)
        )
        #вынеси
        #self.lm_head = nn.Linear(sizeVector, vocabSize)
    def forward(self, x, attn_mask=None):
        # Attention
        h = self.ln1(x)
        z = self.attn(h, h, h, attn_mask=attn_mask)[0]
        x = x + z  # residual

        # FeedForward
        h = self.ln2(x)
        z = self.ff(h)
        x = x + z  # residual

        return x
class TransformerMy(nn.Module):
    def __init__(self, vocabSize=1000,maxLong=100,sizeVector=256, num_layers=6):
        super().__init__()
        self.Vectorization = nn.Embedding(vocabSize, sizeVector)
        self.posEmbed = nn.Embedding(maxLong, sizeVector)
        self.layers = nn.ModuleList([
            TransformerBlock(sizeVector=sizeVector, num_heads=8)
            for _ in range(num_layers)
            ])
        self.lm_head = nn.Linear(sizeVector, vocabSize)
        

    def forward(self, x):
        B, T = x.shape
        
        tok = self.Vectorization(x)
        pos = self.posEmbed(torch.arange(T, device=x.device))
        pos = pos.unsqueeze(0).repeat(B, 1, 1)

        h = tok + pos
        attn_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()


        for layer in self.layers:
            h = layer(h, attn_mask=attn_mask)


        return self.lm_head(h)
    
class BigTransformer(nn.Module):
    def __init__(self, vocabSize=1000, sizeVector=256, num_layers=6, n_models=10):
        super().__init__()
        self.model = TransformerMy(vocabSize=vocabSize,sizeVector=sizeVector,num_layers=num_layers)
        self.blocks = nn.ModuleList([TransformerMy(vocabSize=vocabSize, sizeVector=sizeVector, num_layers=num_layers) 
                                     for _ in range(n_models)])

    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h)
        return h
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

TOKENIZER_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
tokenized = tokenizer
TOKENIZER_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or '[PAD]'
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0
ds = load_dataset("MarkProMaster229/synthetic_dataset")
VOCAB_SIZE = tokenizer.vocab_size
class LoopTraine(nn.Module):
    def tokinizer2(examples):
        MAX_LENGTH = 100#тут неизвестно!
        print(f"Размер словаря: {VOCAB_SIZE}")
        tokenized = tokenizer(
            examples["text"],
            #что это ? 
            truncation=True,
            #зачем нам паддинги 
            padding="max_length",
            #что это 
            max_length=MAX_LENGTH,
            #что это
            return_tensors="pt"
        )
        #это что такое ?
        tokenized["labels"] = tokenized["input_ids"].clone()

        tokenized_dataset = ds.map(
            tokenized,
            batched=True,
            remove_columns=["text"]  # удаляем оригинальный текст
            )
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return tokenized_dataset
        from torch.utils.data import DataLoader
    def Dataloader(self):
        BatchSize = 50
        train_data = DataLoader(
            tokenizer["train"],
            #что это ?
            shuffle=True,
            batch_size=BatchSize,
            drop_last=True 
        )
        #что это ? 
        val_dataloader = DataLoader(
            tokenizer["validation"],
            batch_size=BatchSize,
            drop_last=True
            )   
        
model = BigTransformer(
    vocabSize=VOCAB_SIZE,
    sizeVector=256,
    num_layers=6,
    n_models=10 
    )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

import torch.optim as optim
from tqdm import tqdm  # для прогресс-бара


def lern(self):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    