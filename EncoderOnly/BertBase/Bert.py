import json
import torch
from torch.utils.data import Dataset

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW

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



model_path = "/home/chelovek/bigWork/beartBase"
tokenizer = AutoTokenizer.from_pretrained(model_path)
with open("/home/chelovek/classified_commentsPosiriveOwO.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("/home/chelovek/Рабочий стол/Valid.json","r", encoding="utf-8") as f:
    valJson = json.load(f)

dataset = BertDataset(data, tokenizer)
train_loader = DataLoader(dataset, batch_size=6, shuffle=True)

ValidData = ValidBertDataset(valJson, tokenizer)
valid_loader = DataLoader(ValidData, batch_size = 6, shuffle=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=3,
    torch_dtype="auto",
    device_map="auto"
)

optimizer_grouped_parameters = [
    {
        "params": model.bert.parameters(),
        "lr": 2e-5,
    },
    {
        "params": model.classifier.parameters(),
        "lr": 2e-5,  
    },
]


optimizer = AdamW(optimizer_grouped_parameters)

colVoEpoch = 10

batch = next(iter(train_loader))
allBatch = len(train_loader)


batchValid = next(iter(valid_loader))

print(f"никто никогда не вернется в {allBatch}")
print(batch["input_ids"].shape)
print(batch["labels"])

for ep in range(colVoEpoch):
    model.train()
    TrainLoss = 0
    ValidLoss = 0
    #a = 0
    for batch in train_loader:
        optimizer.zero_grad()
        #a += 1
        #print(f"{a}/{allBatch}")

        outputs = model(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            labels=batch["labels"].to(model.device)
        )

        loss = outputs.loss
        loss.backward()
        
        optimizer.step()

        TrainLoss += loss.item()
        #print(f"номер {train_loader}")

    print(f"Epoch {ep} Train Loss: {TrainLoss / len(train_loader)}")
    model.eval()
    ValidLoss = 0
    for batchV in valid_loader:
    
        with torch.no_grad():
            outputs = model(
                input_ids=batchV["input_ids"].to(model.device),
                attention_mask=batchV["attention_mask"].to(model.device),
                labels=batchV["labels"].to(model.device)
            )
            ValidLoss += outputs.loss.item()
            
    validLoss = ValidLoss / len(valid_loader)
    print(f"Epoch {ep} Loss in Validation data {validLoss}")

#никто никогда не вернется в 1642
#torch.Size([6, 128])
#tensor([0, 0, 0, 0, 1, 1])
#Epoch 0 Train Loss: 0.8680599441028106
#Epoch 0 Loss in Validation data 0.822890071450053
#Epoch 1 Train Loss: 0.7605470823266982
#Epoch 1 Loss in Validation data 0.752548719580109
#Epoch 2 Train Loss: 0.678294919354072
#Epoch 2 Loss in Validation data 0.9361118087293329
#Epoch 3 Train Loss: 0.5917095467470922
#Epoch 3 Loss in Validation data 0.7763154337535033
#Epoch 4 Train Loss: 0.5009688404530522
#Epoch 4 Loss in Validation data 0.9875671009759646
#Epoch 5 Train Loss: 0.4032669892998573
#Epoch 5 Loss in Validation data 1.0399967842810862
#Epoch 6 Train Loss: 0.31500493886783587
#Epoch 6 Loss in Validation data 1.1279511606018688
#Epoch 7 Train Loss: 0.2516790105104526
#Epoch 7 Loss in Validation data 1.1669435783012494
#Epoch 8 Train Loss: 0.1970971176510248
#Epoch 8 Loss in Validation data 1.3306864775163498
#Epoch 9 Train Loss: 0.15996308310330878
#Epoch 9 Loss in Validation data 1.2921368874093466