import json
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.data import Dataset

class BertDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.label2id = {
            "negative": 0,
            "neutral": 1,
            "positive": 2
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text = item["text"]
        label = self.label2id[item["label"]]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }
    
class ValidBertDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.label2id = {
            "negative": 0,
            "neutral": 1,
            "positive": 2
        }
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]

        text = item["text"]
        label = self.label2id[item["label"]]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }
    



tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

model = AutoModelForSequenceClassification.from_pretrained(
    "FacebookAI/xlm-roberta-base",
    num_labels=3,
    torch_dtype="auto",
    device_map="auto"
)


config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules = ["query", "value", "key", "dense"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"]
)

with open("/home/chelovek/classified_commentsPosiriveOwO.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("/home/chelovek/Рабочий стол/Valid.json","r", encoding="utf-8") as f:
    valJson = json.load(f)

dataset = BertDataset(data, tokenizer)


train_loader = DataLoader(dataset, batch_size=12, shuffle=True)

ValidData = ValidBertDataset(valJson, tokenizer)
valid_loader = DataLoader(ValidData, batch_size = 12, shuffle=True)

model = get_peft_model(model, config)
model.print_trainable_parameters()


optimizer = AdamW(model.parameters(), lr=2e-5)
colVoEpoch = 20

import os
roberta_path = "roberta_models"
os.makedirs(roberta_path, exist_ok=True)

for ep in range(colVoEpoch):
    model.train()
    TrainLoss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            labels=batch["labels"].to(model.device)
        )

        loss1 = outputs.loss
        TrainLoss += loss1.item()

        loss1.backward()
        optimizer.step()

    TrainLoss /= len(train_loader)
    print(f"Epoch {ep} loss in train {TrainLoss}")
    if ep % 1 == 0:
        print("save this!")
        model.save_pretrained(f"{roberta_path}/loraForROBERTA_epoch{ep+1}")

    model.eval()
    Valid_loss = 0

    for batch in valid_loader:
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                labels=batch["labels"].to(model.device)
            )
        lossValid = outputs.loss
        Valid_loss += lossValid.item()
    Valid_loss /= len(valid_loader)
    print(f"Epoch {ep} loss in Valid {Valid_loss}")

#Epoch 0 loss in train 0.9158469306783757
#save this!
#Epoch 0 loss in Valid 0.7410750765549509
#Epoch 1 loss in train 0.737323088528323
#save this!
#Epoch 1 loss in Valid 0.6086877993258991
#Epoch 2 loss in train 0.6775894329115186
#save this!
#Epoch 2 loss in Valid 0.5845026852268922
#Epoch 3 loss in train 0.643513333891835
#save this!
#Epoch 3 loss in Valid 0.5461553405774268
#Epoch 4 loss in train 0.6184801802529197
#save this!
#Epoch 4 loss in Valid 0.5396700407329359
#Epoch 5 loss in train 0.5931856373050471
#save this!
#Epoch 5 loss in Valid 0.5236105121868221
#Epoch 6 loss in train 0.5723478227272975
#save this!
#Epoch 6 loss in Valid 0.5343811045351782
#Epoch 7 loss in train 0.553374152562505
#save this!
#Epoch 7 loss in Valid 0.574058389977405
#Epoch 8 loss in train 0.5281864021622307
#save this!
#Epoch 8 loss in Valid 0.5165252160084876
#Epoch 9 loss in train 0.5149054350592333
#save this!
#Epoch 9 loss in Valid 0.6286481352228868
#Epoch 10 loss in train 0.49756711899290423
#save this!
#Epoch 10 loss in Valid 0.5128102786839008
#Epoch 11 loss in train 0.48466512180601634
#save this!
#Epoch 11 loss in Valid 0.5384572548301596
#Epoch 12 loss in train 0.47197424308663455
#save this!
#Epoch 12 loss in Valid 0.587054130645763
#Epoch 13 loss in train 0.44659905255875404
#save this!
#Epoch 13 loss in Valid 0.5688826889289837
#Epoch 14 loss in train 0.4435478064495884
#save this!
#Epoch 14 loss in Valid 0.8040276240361365
#Epoch 15 loss in train 0.4270620637853493
#save this!
#Epoch 15 loss in Valid 0.6674159624074635
#Epoch 16 loss in train 0.41355355001132976
#save this!
#Epoch 16 loss in Valid 0.6027880973721805
#Epoch 17 loss in train 0.4019455197749344
#save this!
#Epoch 17 loss in Valid 0.5945103256718108
#Epoch 18 loss in train 0.3846266034595786
#save this!
#Epoch 18 loss in Valid 0.954741687366837
#Epoch 19 loss in train 0.373605940382061
#save this!
#Epoch 19 loss in Valid 0.6239002806771743