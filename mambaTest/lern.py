import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler      # ← + GradScaler
from transformers import AutoTokenizer

from mambaMain import build_model
from data import DataModule, load_json_data


# ─── Токенизатор ───
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token
print("vocab_size:", len(tokenizer))

device = torch.device("cuda")

# ─── Данные ───
data = load_json_data("/run/media/user/Storage/ml-lub/ml-lab/MydatasetT2_F.json")
print(f"Всего примеров в JSON: {len(data)}")

dm = DataModule(
    data=data,
    tokenizer=tokenizer,
    seq_len=512,
    batch_size=16,
    val_ratio=0.1,
    num_workers=2,
)
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()

# ─── Модель ───
vocab_size = len(tokenizer)
model = build_model(vocab_size, config={
    "d_model": 512,
    "max_len": 1024,
    "n_heads_q": 16,
    "n_heads_kv": 4,
    "head_dim": 64,
    "dropout": 0.1,
}).to(device)
model.lm_head.weight = model.token_embedding.weight

# ─── Optimizer + Scaler ───
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()                 # ← для fp16
amp_dtype = torch.float16             # ← было bfloat16

# ─── Проверка батча ───
batch = next(iter(train_loader))
print("input_ids:", batch["input_ids"].shape)
print("labels:   ", batch["labels"].shape)
real = (batch["labels"][0] != -100).sum().item()
total = batch["labels"][0].numel()
print(f"Реальных токенов в примере 0: {real}/{total} ({100*real/total:.1f}%)")

# ─── Обучение ───
num_epochs = 10
for epoch in range(num_epochs):
    # === TRAIN ===
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast(dtype=amp_dtype):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()                          # ←
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)                                 # ←
        scaler.update()                                        # ←

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # === VALIDATION ===
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast(dtype=amp_dtype):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    labels.view(-1),
                    ignore_index=-100,
                )
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"epoch {epoch+1:3d} | train_loss {train_loss:.4f} (ppl {math.exp(train_loss):.2f}) "
          f"| val_loss {val_loss:.4f} (ppl {math.exp(val_loss):.2f})")

torch.save(model.state_dict(), "final_model.pt")