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

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
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

tokenizer = AutoTokenizer.from_pretrained(model_path)

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
        model.save_pretrained(f"loraForBERT{ep+1}")

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



#Epoch 0 loss in train 0.9268037862667358
#save this!
#Epoch 0 loss in Valid 0.9856747420210588
#Epoch 1 loss in train 0.8913424615447617
#save this!
#Epoch 1 loss in Valid 0.9158578201344139
#Epoch 2 loss in train 0.8698545825031294
#save this!
#Epoch 2 loss in Valid 0.8792771916640433
#Epoch 3 loss in train 0.8371046626901801
#save this!
#Epoch 3 loss in Valid 0.7641685777588895
#Epoch 4 loss in train 0.8092562085818431
#save this!
#Epoch 4 loss in Valid 0.8229999385382
#Epoch 5 loss in train 0.7869190592205336
#save this!
#Epoch 5 loss in Valid 0.7866415820623699
#Epoch 6 loss in train 0.7665394056366074
#save this!
#Epoch 6 loss in Valid 0.8540124422625491
#Epoch 7 loss in train 0.749155752732896
#save this!
#Epoch 7 loss in Valid 0.7070749998092651
#Epoch 8 loss in train 0.7298482119601478
#save this!
#Epoch 8 loss in Valid 0.720144293810192
#Epoch 9 loss in train 0.7147667670075699
#save this!
#Epoch 9 loss in Valid 0.6904838006747397
#Epoch 10 loss in train 0.7015375231237272
#save this!
#Epoch 10 loss in Valid 0.7583794185989782
#Epoch 11 loss in train 0.6888231706604743
#save this!
#Epoch 11 loss in Valid 0.8473481159461173
#Epoch 12 loss in train 0.6744434226045655
#save this!
#Epoch 12 loss in Valid 0.6963813351957422
#Epoch 13 loss in train 0.6713746627667935
#save this!
#Epoch 13 loss in Valid 0.6966002458020261
#Epoch 14 loss in train 0.6493793887684028
#save this!
#Epoch 14 loss in Valid 0.6894897898953212
#Epoch 15 loss in train 0.6380676526063542
#save this!
#Epoch 15 loss in Valid 0.7037043759697362
#Epoch 16 loss in train 0.6354927422431151
#save this!
#Epoch 16 loss in Valid 0.6870511283999995
#Epoch 17 loss in train 0.6155846477752771
#save this!
#Epoch 17 loss in Valid 0.7176429503842404
#Epoch 18 loss in train 0.6124847058329309
#save this!
#Epoch 18 loss in Valid 0.7578575469945606
#Epoch 19 loss in train 0.5969803444269804
#save this!
#Epoch 19 loss in Valid 0.824189891940669