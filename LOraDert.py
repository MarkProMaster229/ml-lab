import json
import torch
from torch.utils.data import Dataset
class DatasetGiv:

    def __init__(self, json_path, tokenizer, max_length=100):
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
        
        prompt = f"<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n"
        target = f"{answer}<|im_end|>"
        full_text = prompt + target

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
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "labels": labels 
            }


class DatasetValid:
        def __init__(self, json_path, tokenizer, max_length=100):
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
            
            prompt = f"<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n"
            target = f"{answer}<|im_end|>"
            full_text = prompt + target


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
            labels[labels == tokenizer.pad_token_id] = -100

            return {
                "input_ids": input_ids, 
                "attention_mask": attention_mask, 
                "labels": labels 
                }
import torch
import os
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


max_memory_config = {0: "7GB", "cpu": "20GB"}
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)


model_path = "/home/chelovek/ML/modelFull/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory=max_memory_config
)




config = LoraConfig(
    r=20,
    lora_alpha=40,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()


model.config.use_cache = False


dataset = DatasetGiv("/home/chelovek/Downloads/train_dataset_filtered.json", tokenizer)
ValidDataset = DatasetValid("/home/chelovek/Downloads/valid_dataset_filtered.json", tokenizer)
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "/home/chelovek/ML/modelFull/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)


training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=6,
    num_train_epochs=8,
    fp16=True,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
    optim="adamw_torch",
    report_to="none",
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    logging_first_step=True,
    remove_unused_columns=False,
)

print("\n1234")
batch = dataset[0]
input_ids = batch["input_ids"].unsqueeze(0).to(model.device)
attention_mask = batch["attention_mask"].unsqueeze(0).to(model.device)
labels = batch["labels"].unsqueeze(0).to(model.device)
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
print(f"Loss: {outputs.loss.item():.4f}, requires_grad: {outputs.loss.requires_grad}")


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=ValidDataset,
)


trainer.train()


model.save_pretrained("./final_lora_adapter2")
tokenizer.save_pretrained("./final_lora_adapter2")
print("Обучение завершено")