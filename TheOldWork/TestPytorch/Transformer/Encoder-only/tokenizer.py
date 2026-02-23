import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

class TokenizerForClassification():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        self.ds = load_dataset("json", data_files="/home/chelovek/Загрузки/combined_dataset_with_augmented_pairs.json")
        
        self.label_map = {
            "negative": 0,
            "positive": 1,
            "neutral": 2
        }
    
    def tokenize(self, examples):
        label_str = examples["label"]
        label_num = self.label_map[label_str]
        
        encoded = self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=100,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_num)
        }
    
    def get_vocab_size(self):
        return self.tokenizer.vocab_size
    
    def prepare_dataset(self):
        dataset = self.ds.map(
            self.tokenize, 
            batched=False,
            remove_columns=["text", "label", "route_url", "page"]
        )
        
        tokenized_dataset = dataset.with_format(
            "torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        return tokenized_dataset
    
    def dataloader(self):
        dataset = self.prepare_dataset()
        train_data = dataset["train"]
        train_loader = DataLoader(
            train_data,
            batch_size=26,
            shuffle=True
        )
        return train_loader