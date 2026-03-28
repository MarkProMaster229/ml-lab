import json
import torch
from torch.utils.data import Dataset
class DatasetGiv:

    def __init__(self, json_path, tokenizer, max_length=128):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): 
        return len(self.data) 
    def __getitem__(self, idx): 
        item = self.data[idx] 

        question = item["question"] 
        answer = item["answer"]
        text = f"{question} {answer}"

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt" 
            )
        input_ids = encoding["input_ids"].squeeze(0) 
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "labels": labels 
            }
    
class DatasetValid:
        def __init__(self, json_path, tokenizer, max_length=128):
            with open(json_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self): 
            return len(self.data)
        
        def __getitem__(self, idx): 
            item = self.data[idx] 

            question = item["question"] 
            answer = item["answer"]
            text = f"{question} {answer}"

            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt" 
                )
            input_ids = encoding["input_ids"].squeeze(0) 
            attention_mask = encoding["attention_mask"].squeeze(0)
            labels = input_ids.clone()

            labels[labels == self.tokenizer.pad_token_id] = -100

            return {
                "input_ids": input_ids, 
                "attention_mask": attention_mask, 
                "labels": labels 
                }


from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
model_path = "/home/chelovek/bigWork/ModelArch"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
for name, module in model.named_modules():
    print(name)
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules = ["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)

dataset = DatasetGiv("/home/chelovek/Загрузки/Telegram Desktop/myJson.json", tokenizer)
ValidDataset = DatasetValid("/home/chelovek/Рабочий стол/ValTool.json", tokenizer)
print("eos_token:", tokenizer.eos_token)
print("pad_token:", tokenizer.pad_token)

print("eos_token_id:", tokenizer.eos_token_id)
print("pad_token_id:", tokenizer.pad_token_id)
train_loader = DataLoader(
    dataset,
    batch_size=6,
    shuffle=True
)
validDataset = DataLoader(
    ValidDataset,
    batch_size=6,
    shuffle=True
)




model = get_peft_model(model, config)
model.print_trainable_parameters()
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=1e-4)
colVoEpoch = 10
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
    print("loss in train")
    print(TrainLoss)
    if ep % 1 == 0:
        print("save this!")
        model.save_pretrained(f"loraForArch{ep+1}")

    model.eval()
    Valid_loss = 0
    for batch in validDataset:
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                labels=batch["labels"].to(model.device)
            )
        lossValid = outputs.loss
        Valid_loss += lossValid.item()
    Valid_loss /= len(validDataset)
    print("loss in Valid")
    print(Valid_loss)