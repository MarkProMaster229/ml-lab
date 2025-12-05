import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader



class TokenizerMy():
    def __init__(self):
        # Выносим инициализацию в __init__
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        self.ds = load_dataset("MarkProMaster229/synthetic_dataset")

    def tokenize(self, examples):
        result = self.tokenizer(
        examples["target"],
        truncation=True,
        padding="max_length",
        max_length=100,
        return_tensors="pt"
        )
        #вот тут подробнее почему именно так ? 
        result["labels"] = result["input_ids"].clone()
        return result
    
    def get_vocab_size(self):
        
        return self.tokenizer.vocab_size
    def tokenizerOutout(self):
        tokenizeOutput = self.ds.map(self.tokenize, batched=True, remove_columns=["input", "target"])
        tokenizeOutput = tokenizeOutput.with_format("torch", columns=["input_ids", "labels"])
        return tokenizeOutput
    
    def datalouder(self):
        trainLoader = DataLoader(
            self.tokenizerOutout()["train"],
            batch_size=24,
            #попробуй без перемешивания потом
            shuffle=True,
        )
        return trainLoader
    
    