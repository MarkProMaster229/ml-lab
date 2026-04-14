import json
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import torch
model_path = "/home/chelovek/ModelFlaffyTail_Quantized_int4"

tokenizer = AutoTokenizer.from_pretrained(model_path)

class DatasetTrain:
        def __init__(self, json_path, tokenizer, max_length=110):
            with open(json_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self): 
            return len(self.data)
                
        def __getitem__(self, idx): 
            item = self.data[idx] 
            
            question = item["input"] 
            answer = item["target"]
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
            
            prompt_messages = [{"role": "user", "content": question}]
            prompt = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt" 
            )
            input_ids = encoding["input_ids"].squeeze(0) 
            attention_mask = encoding["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            
            prompt_encoding = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            prompt_len = prompt_encoding["input_ids"].shape[1]
            
            labels[:prompt_len] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100

            return {
                "input_ids": input_ids, 
                "attention_mask": attention_mask, 
                "labels": labels 
            }

class DatasetValid:
        def __init__(self, json_path, tokenizer, max_length=110):
            with open(json_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self): 
            return len(self.data)
                
        def __getitem__(self, idx): 
            item = self.data[idx] 
            
            question = item["input"] 
            answer = item["target"]
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
            
            prompt_messages = [{"role": "user", "content": question}]
            prompt = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt" 
            )
            input_ids = encoding["input_ids"].squeeze(0) 
            attention_mask = encoding["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            
            prompt_encoding = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            prompt_len = prompt_encoding["input_ids"].shape[1]
            
            labels[:prompt_len] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                "input_ids": input_ids, 
                "attention_mask": attention_mask, 
                "labels": labels 
            }
        

model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype="auto",
    device_map="auto"
)
for name, module in model.named_modules():
    print(name)
config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules = ["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)
datasetTrain = DatasetTrain("/home/chelovek/Видео/train_dataset22222.json",tokenizer)
datasetValid = DatasetValid("/home/chelovek/Видео/val_dataset2222222.json",tokenizer)



train_loader = DataLoader(
    datasetTrain,
    batch_size=3,
    shuffle=True
)
validDataset = DataLoader(
    datasetValid,
    batch_size=3,
    shuffle=True
)

print(train_loader)

model = get_peft_model(model, config)
model.print_trainable_parameters()
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=3e-5)
colVoEpoch = 10
accumulation_steps = 3

for ep in range(colVoEpoch):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(train_loader):
        outputs = model(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            labels=batch["labels"].to(model.device)
        )
        
        loss = outputs.loss / accumulation_steps
        loss.backward()
        
        total_loss += outputs.loss.item()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {ep+1}, Average Loss: {avg_loss:.4f}")
    
    model.save_pretrained(f"loraForFlaffyTail{ep+1}")

    model.eval()
    total_valid_loss = 0
    total_valid_samples = 0
    for batch in validDataset:
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                labels=batch["labels"].to(model.device)
            )
        batch_loss = outputs.loss.item()
        batch_size = batch["input_ids"].size(0)
        total_valid_loss += batch_loss * batch_size
        total_valid_samples += batch_size
    Valid_loss = total_valid_loss / total_valid_samples
    print("loss in Valid")
    print(Valid_loss)