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
roberta_path = "BertBaseFull"

optimizer = AdamW(optimizer_grouped_parameters)

colVoEpoch = 20

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
    if ep % 1 == 0:
        #print("save this!")
        model.save_pretrained(f"{roberta_path}/BertBaseFullModel{ep+1}")

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
#Epoch 0 Train Loss: 0.8686387965253732
#Epoch 0 Loss in Validation data 0.769134180771338
#Epoch 1 Train Loss: 0.7611743432405007
#Epoch 1 Loss in Validation data 0.746833496802562
#Epoch 2 Train Loss: 0.675213236938005
#Epoch 2 Loss in Validation data 0.7892871507116266
#Epoch 3 Train Loss: 0.595656931611293
#Epoch 3 Loss in Validation data 0.7977193585420782
#Epoch 4 Train Loss: 0.49758275126954415
#Epoch 4 Loss in Validation data 0.887513331464819
#Epoch 5 Train Loss: 0.3946617727069214
#Epoch 5 Loss in Validation data 0.9465082672399443
#Epoch 6 Train Loss: 0.3112003424565901
#Epoch 6 Loss in Validation data 1.100050000247319
#Epoch 7 Train Loss: 0.2355832863290937
#Epoch 7 Loss in Validation data 1.274325226232208
#Epoch 8 Train Loss: 0.18996810883570897
#Epoch 8 Loss in Validation data 1.1774581715659667
#Epoch 9 Train Loss: 0.15845887185246388
#Epoch 9 Loss in Validation data 1.3258019001097292
#Epoch 10 Train Loss: 0.12964913105513717
#Epoch 10 Loss in Validation data 1.6574379938679773
#Epoch 11 Train Loss: 0.1184352965175647
#Epoch 11 Loss in Validation data 1.5163718392267018
#Epoch 12 Train Loss: 0.10106043583907132
#Epoch 12 Loss in Validation data 1.6204950932532902
#Epoch 13 Train Loss: 0.09975709741751577
#Epoch 13 Loss in Validation data 1.499698380192402
#Epoch 14 Train Loss: 0.093336668817841
#Epoch 14 Loss in Validation data 1.3822113348617897
#Epoch 15 Train Loss: 0.08832583144544044
#Epoch 15 Loss in Validation data 1.6743186236031957
#Epoch 16 Train Loss: 0.07559707110726638
#Epoch 16 Loss in Validation data 1.8559099168161157
#Epoch 17 Train Loss: 0.07997204229968315
#Epoch 17 Loss in Validation data 1.849636017370063
#Epoch 18 Train Loss: 0.07319258857468558
#Epoch 18 Loss in Validation data 1.7581849822623503
#Epoch 19 Train Loss: 0.06875287767275788
#Epoch 19 Loss in Validation data 1.6340184822879933