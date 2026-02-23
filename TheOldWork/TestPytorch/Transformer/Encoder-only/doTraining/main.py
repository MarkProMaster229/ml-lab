import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import os
from transformers import AutoTokenizer
from datasets import load_dataset
import random

class TransformerBlock(nn.Module):
    def __init__(self, sizeVector=256, numHeads=8, dropout=0.5):
        super().__init__()
        self.ln1 = nn.LayerNorm(sizeVector)
        self.attn = nn.MultiheadAttention(sizeVector, numHeads, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(sizeVector)
        self.ff = nn.Sequential(
            nn.Linear(sizeVector, sizeVector*4),
            nn.GELU(),
            nn.Linear(sizeVector*4, sizeVector),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, attention_mask=None):
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + self.dropout_attn(attn_out)
        x = x + self.ff(self.ln2(x))
        return x


class TransformerRun(nn.Module):
    def __init__(self, vocabSize=120000, maxLen=100, sizeVector=256, numBlocks=4, numHeads=8, numClasses=3, dropout=0.5):
        super().__init__()
        self.token_emb = nn.Embedding(vocabSize, sizeVector)
        self.pos_emb = nn.Embedding(maxLen, sizeVector)
        self.layers = nn.ModuleList([
            TransformerBlock(sizeVector=sizeVector, numHeads=numHeads, dropout=dropout)
            for _ in range(numBlocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(sizeVector*2)
        self.classifier = nn.Linear(sizeVector*2, numClasses)

    def forward(self, x, attention_mask=None):
        B, T = x.shape
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0).expand(B, T))
        h = tok + pos

        for layer in self.layers:
            h = layer(h, attention_mask)

        cls_token = h[:,0,:]   # CLS
        mean_pool = h.mean(dim=1)
        combined = torch.cat([cls_token, mean_pool], dim=1)
        combined = self.ln(self.dropout(combined))
        logits = self.classifier(combined)
        return logits

    


class TokenizerForClassification():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        self.ds = load_dataset("json", data_files="/home/chelovek/Рабочий стол/telegramParsClass.json")
        
        self.label_map = {
            "positive":0, 
            "negative":1, 
            "neutral":2
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

def save_model(model, tokenizer, config, output_dir, label_map=None):
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(output_dir, 'model_weights.pth'))
    
    full_config = {
        'vocabSize': config['vocabSize'],
        'maxLong': config['maxLong'],
        'sizeVector': config['sizeVector'],
        'numLayers': config['numLayers'],
        'numHeads': 8,
        'numClasses': 3,
    }
    torch.save(full_config, os.path.join(output_dir, 'config.pth'))
    
    if hasattr(tokenizer, 'save_pretrained'):
        tokenizer.save_pretrained(output_dir)

    if label_map is not None:
        with open(os.path.join(output_dir, 'label_map.json'), 'w', encoding='utf-8') as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)

def finetune_with_json():
    
    model_dir = '/home/chelovek/Downloads/modelClass/'
    data_path = '/home/chelovek/work/balanced_output00000.json'
    #this configuration 
    config_path = os.path.join(model_dir, 'config.pth')
    config = torch.load(config_path, map_location='cpu')
    #weigh model) 
    weights_path = os.path.join(model_dir, 'model_weights.pth')
    weights = torch.load(weights_path, map_location='cpu')
    
    model = TransformerRun(
    vocabSize=config['vocabSize'],
    maxLen=config['maxLong'],
    sizeVector=config['sizeVector'],
    numBlocks=config['numLayers']
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
            self.label_map = {'POSITIVE': 0, 'NEGATIVE': 1, 'NEUTRAL': 2}
            
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            text = item['text']
            label_str = item['label']

            tokens = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
                )
            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)

            label = self.label_map.get(label_str.upper(), 2)

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': torch.tensor(label, dtype=torch.long)
                }

    

    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    train_dataset = JSONDataset(train_data, tokenizer, max_length=config['maxLong'])
    val_dataset = JSONDataset(val_data, tokenizer, max_length=config['maxLong'])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\ndevice: {device}")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 2

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)

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
                
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"\nEpoch {epoch+1}/{num_epochs}:")

        print(f"  Train - Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val - Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")


    output_dir = '/home/chelovek/Документы/work/classifier7772finalycutBIIIGBOOOOS299'
    save_model(
        model=model,
        tokenizer=tokenizer,
        config=config,
        output_dir=output_dir,
        label_map={'positive': 0, 'negative': 1, 'neutral': 2}
    )

if __name__ == "__main__":
    finetune_with_json()