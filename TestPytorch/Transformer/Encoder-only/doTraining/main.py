import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import os
from transformers import AutoTokenizer
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

class TransformerBlock(nn.Module):
    def __init__(self, sizeVector=128, numHeads=4):
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
    def __init__(self, vocabSize=120000, maxLong=100, sizeVector=128, block=4):
        super().__init__()
        self.maxLong = maxLong 
        self.Vectorization = nn.Embedding(vocabSize, sizeVector)
        self.posEmbed = nn.Embedding(maxLong, sizeVector)
        self.layers = nn.ModuleList([
            TransformerBlock(sizeVector=sizeVector, numHeads=8)
            for _ in range(block)
        ])
        self.lmHead = nn.Linear(sizeVector, 3)
    
    def forward(self, x):
        B, T = x.shape
        tok = self.Vectorization(x)
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pos = self.posEmbed(positions)
        h = tok + pos
        
        for layer in self.layers:
            h = layer(h)
        
        cls = h.mean(dim=1)
        logits = self.lmHead(cls)
        return logits
    


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
    

def finetune_with_json():
    
    model_dir = '/home/chelovek/Документы/work/classifier_epoch70'
    data_path = '/home/chelovek/Загрузки/en_ru/synthetic_classification_dataset(1).json'
    #this configuration 
    config_path = os.path.join(model_dir, 'config.pth')
    config = torch.load(config_path, map_location='cpu')
    #weigh model) 
    weights_path = os.path.join(model_dir, 'model_weights.pth')
    weights = torch.load(weights_path, map_location='cpu')
    
    model = TransformerRun(
        vocabSize=config['vocabSize'],
        maxLong=config['maxLong'],
        sizeVector=config['sizeVector'],
        block=config['numLayers']
    )
    model.load_state_dict(weights)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    with open(data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    print(f"dowlound {len(all_data)} test kit")

    class JSONDataset(Dataset):
        def __init__(self, data, tokenizer, max_length=100):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
            
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            text = item['text']
            label_str = item['label']

            tokens = tokenizer.encode(text)
            tokens = tokens[:self.max_length]
            if len(tokens) < self.max_length:
                tokens = tokens + [0] * (self.max_length - len(tokens))
            input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
            
            label = self.label_map.get(label_str, 2) 

            return {
                'input_ids': input_ids.squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long)
            }
    

    import random
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    train_dataset = JSONDataset(train_data, tokenizer, max_length=config['maxLong'])
    val_dataset = JSONDataset(val_data, tokenizer, max_length=config['maxLong'])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\ndevice: {device}")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 12

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"\nEpoch {epoch+1}/{num_epochs}:")

        print(f"  Train - Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val - Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")


    output_dir = '/home/chelovek/Документы/work/classifier'
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model_weights.pth'))
    torch.save(config, os.path.join(output_dir, 'config.pth'))

    if hasattr(tokenizer, 'save_pretrained'):
        tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    finetune_with_json()