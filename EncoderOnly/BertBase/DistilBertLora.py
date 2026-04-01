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
    



tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased",
    num_labels=3,
    torch_dtype="auto",
    device_map="auto"
)


config = LoraConfig(
    r=8,
    lora_alpha=16,
    #target_modules = ["query", "value", "key", "dense"],
    target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"],
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
roberta_path = "distil_Bert"
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
        model.save_pretrained(f"{roberta_path}/loraForDistil_Bert_epoch{ep+1}")

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

#Epoch 0 loss in train 0.9144839738086138
#save this!
#Epoch 0 loss in Valid 0.9012686955301386
#Epoch 1 loss in train 0.8720600287022747
#save this!
#Epoch 1 loss in Valid 0.8137206654799612
#Epoch 2 loss in train 0.8450387906505478
#save this!
#Epoch 2 loss in Valid 0.7938372963353207
#Epoch 3 loss in train 0.8304082767609121
#save this!
#Epoch 3 loss in Valid 0.7822490585477728
#Epoch 4 loss in train 0.8218647722930769
#save this!
#Epoch 4 loss in Valid 0.7740178743475362
#Epoch 5 loss in train 0.8122451209194332
#save this!
#Epoch 5 loss in Valid 0.8751911238620156
#Epoch 6 loss in train 0.8030335002927048
#save this!
#Epoch 6 loss in Valid 0.7790471315383911
#Epoch 7 loss in train 0.7998993278420944
#save this!
#Epoch 7 loss in Valid 0.7613500482157657
#Epoch 8 loss in train 0.7917967596602934
#save this!
#Epoch 8 loss in Valid 0.7891154101020411
#Epoch 9 loss in train 0.7772026442799005
#save this!
#Epoch 9 loss in Valid 0.8104652825154757
#Epoch 10 loss in train 0.7748061650853499
#save this!
#Epoch 10 loss in Valid 0.7402110601726332
#Epoch 11 loss in train 0.7684638743423806
#save this!
#Epoch 11 loss in Valid 0.7508765317891773
#Epoch 12 loss in train 0.7598204258443667
#save this!
#Epoch 12 loss in Valid 0.7105655403513658
#Epoch 13 loss in train 0.7526728668761747
#save this!
#Epoch 13 loss in Valid 0.7288588865807182
#Epoch 14 loss in train 0.7470956186284101
#save this!
#Epoch 14 loss in Valid 0.7330299913883209
#Epoch 15 loss in train 0.7375644864321627
#save this!
#Epoch 15 loss in Valid 0.751100701721091
#Epoch 16 loss in train 0.732530345870284
#save this!
#Epoch 16 loss in Valid 0.769866921399769
#Epoch 17 loss in train 0.7242152105005011
#save this!
#Epoch 17 loss in Valid 0.7093124938638586
#Epoch 18 loss in train 0.7182778385757094
#save this!
#Epoch 18 loss in Valid 0.7614692888761821
#Epoch 19 loss in train 0.7149477876568537
#save this!
#Epoch 19 loss in Valid 0.7259659908319774