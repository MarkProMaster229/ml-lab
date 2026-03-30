import json
import torch
from torch.utils.data import Dataset

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
model_path = "/home/chelovek/bigWork/beartBase"
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
optimizer = AdamW(model.parameters(), lr=1e-4)

colVoEpoch = 10
for ep in range(colVoEpoch):
    model.train()
    TrainLoss = 0
    for batch in train_loader:
        optimizer.zero_grad()