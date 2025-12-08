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
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        self.ds = load_dataset("json", data_files="/home/chelovek/Музыка/dataset.json")
    def tokenize(self, examples):
        input = self.tokenizer(
            examples["input"],
            truncation=True,
            padding="max_length",
            max_length=200,
            return_tensors="pt"
        )

        target = self.tokenizer(
            examples["target"],
            truncation=True,
            padding="max_length",
            max_length=200,
            return_tensors="pt"
        )
        return {
            "input_ids": input["input_ids"],
            "attention_mask": input["attention_mask"],
            "labels": target["input_ids"]
        }
    def get_vocab_size(self):
        return self.tokenizer.vocab_size
    
    def NotSplit(self):
        notsplit = self.ds.map(
            self.tokenize, 
            batched=True,
            remove_columns=["input", "target"]
            )
        tokenized_dataset = notsplit.with_format(
            "torch",
            columns=["input_ids", "attention_mask", "labels"]
          )
        return tokenized_dataset
    
    def datalouder(self):
        dataset = self.NotSplit()
        train_data = dataset["train"]
        trainLoader = DataLoader(
            train_data,
            batch_size=16,
            shuffle=True
        )
        return trainLoader

     