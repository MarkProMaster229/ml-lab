import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
ds = load_dataset("MarkProMaster229/synthetic_dataset")


class TokenizerMy():

    def tokenize(self, examples):
        result = tokenizer(
        examples["target"],
        truncation=True,
        padding="max_length",
        max_length=100,
        return_tensors="pt"
        )
        #вот тут подробнее почему именно так ? 
        result["labels"] = result["input_ids"].clone()
        return result
    def tokenizerOutout(self):
        tokenizeOutput = ds.map(self.tokenize, batched=True, remove_columns=["input", "target"])
        tokenizeOutput = tokenizeOutput.with_format("torch", columns=["input_ids", "labels"])
        return tokenizeOutput
    
    def datalouder(self):
        trainLoader = DataLoader(
            self.tokenizerOutout()["train"],
            batch_size=46,
            #попробуй без перемешивания потом
            shuffle=True,
        )
        return trainLoader
    
    