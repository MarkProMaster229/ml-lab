import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from safetensors.torch import save_file, load_file
import torch.nn as nn
from tqdm import tqdm

import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer

from mambaMain import build_model
from data import DataModule, load_json_data
from pathlib import Path


# ─── Токенизатор ───
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token
print("vocab_size:", len(tokenizer))

device = torch.device("cuda")

# ─── Данные ───
data = load_json_data("/run/media/user/Storage/ml-lub/valid.json")
print(f"Всего примеров в JSON: {len(data)}")


def save_checkpoint(model, optimizer, epoch, global_step, config, ckpt_dir, best_val_loss=None):
    """Сохраняет модель в safetensors, оптимизатор + мета в .pt"""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # 1) Модель → safetensors
    save_file(model.state_dict(), ckpt_dir / "model.safetensors")
    
    # 2) Optimizer + метаданные → .pt
    meta = {
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
    }
    if best_val_loss is not None:
        meta["best_val_loss"] = best_val_loss
    torch.save(meta, ckpt_dir / "trainer_state.pt")
    
    print(f"  → saved {ckpt_dir}/model.safetensors + trainer_state.pt")


def load_checkpoint(model, optimizer, ckpt_dir, device):
    """Загружает модель из safetensors, оптимизатор + мета из .pt"""
    ckpt_dir = Path(ckpt_dir)
    
    # 1) Модель
    state_dict = load_file(ckpt_dir / "model.safetensors", device=str(device))
    model.load_state_dict(state_dict)
    
    # 2) Optimizer + мета
    meta = torch.load(ckpt_dir / "trainer_state.pt", map_location=device)
    if optimizer is not None:
        optimizer.load_state_dict(meta["optimizer_state"])
    
    epoch = meta.get("epoch", 0)
    global_step = meta.get("global_step", 0)
    best_val_loss = meta.get("best_val_loss", float("inf"))
    print(f"  ← loaded {ckpt_dir} (epoch {epoch}, step {global_step})")
    return epoch, global_step, best_val_loss


# ─── Данные (DataModule) ───
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
# model.lm_head.weight = model.token_embedding.weight   # tying


# ─── ПРАВИЛЬНАЯ ИНИЦИАЛИЗАЦИЯ ───
def init_gpt_weights(module):
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# Применяем ко всему, КРОМЕ Mamba
for name, module in model.named_modules():
    if "mamba" in name:
        continue
    init_gpt_weights(module)

# Отдельно — embedding и position_embedding
nn.init.normal_(model.token_embedding.weight, mean=0.0, std=0.02)
nn.init.normal_(model.position_embedding.weight, mean=0.0, std=0.02)


# ─── ПРОВЕРКА ЛОГИТОВ ───
model.eval()
with torch.no_grad():
    batch = next(iter(train_loader))
    x = batch["input_ids"].to(device)
    logits = model(x)
    print(f"logits min:  {logits.min().item():.3f}")
    print(f"logits max:  {logits.max().item():.3f}")
    print(f"logits mean: {logits.mean().item():.3f}")
    print(f"logits std:  {logits.std().item():.3f}")
model.train()


# ─── Optimizer + Scaler ───
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()
amp_dtype = torch.float16


# ─── Проверка батча ───
batch = next(iter(train_loader))
print("input_ids:", batch["input_ids"].shape)
print("labels:   ", batch["labels"].shape)
real = (batch["labels"][0] != -100).sum().item()
total = batch["labels"][0].numel()
print(f"Реальных токенов в примере 0: {real}/{total} ({100*real/total:.1f}%)")


# ─── Config ───
config = {
    "vocab_size": vocab_size,
    "d_model": 512,
    "max_len": 1024,
    "n_heads_q": 16,
    "n_heads_kv": 4,
    "head_dim": 64,
    "dropout": 0.1,
}


# ─── Папки для чекпоинтов ───
ckpt_dir = Path("checkpoints")
ckpt_dir.mkdir(exist_ok=True)
(ckpt_dir / "tokenizer").mkdir(exist_ok=True)

# Папка для промежуточных чекпоинтов по шагам
step_ckpt_dir = Path("/home/user/save")
step_ckpt_dir.mkdir(parents=True, exist_ok=True)

tokenizer.save_pretrained(ckpt_dir / "tokenizer")
print(f"Tokenizer saved to {ckpt_dir / 'tokenizer'}")


# ─── Обучение ───
num_epochs = 50
save_every_n_epochs = 1
save_every_n_steps = 100000        # промежуточное сохранение по шагам
resume_from = None                 # "checkpoints/last" для продолжения
global_step = 0
best_val_loss = float("inf")
start_epoch = 0

if resume_from is not None:
    start_epoch, global_step, best_val_loss = load_checkpoint(
        model, optimizer, resume_from, device
    )
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir / "tokenizer")


for epoch in range(start_epoch, num_epochs):
    # === TRAIN ===
    model.train()
    train_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast(dtype=amp_dtype):
            logits = model(input_ids)

        logits = logits.float()
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        global_step += 1
        
        # Обновляем прогресс-бар каждые 10 батчей (реже = быстрее)
        if global_step % 10 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # ─── Промежуточное сохранение по шагам ───
        if global_step % save_every_n_steps == 0:
            save_checkpoint(
                model, optimizer, epoch + 1, global_step, config,
                step_ckpt_dir / "last", best_val_loss
            )

    train_loss /= len(train_loader)
    pbar.close()

    # === VALIDATION ===
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]", leave=False):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast(dtype=amp_dtype):
                logits = model(input_ids)

            logits = logits.float()
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"epoch {epoch+1:3d} | train_loss {train_loss:.4f} (ppl {math.exp(train_loss):.2f}) "
          f"| val_loss {val_loss:.4f} (ppl {math.exp(val_loss):.2f})")

    # ─── СОХРАНЕНИЕ ПО ЭПОХАМ ───
    # 1) last — всегда (для resume)
    save_checkpoint(
        model, optimizer, epoch + 1, global_step, config,
        ckpt_dir / "last", best_val_loss
    )

    # 2) Периодически (для истории)
    if (epoch + 1) % save_every_n_epochs == 0:
        save_checkpoint(
            model, optimizer, epoch + 1, global_step, config,
            ckpt_dir / f"epoch_{epoch+1:03d}"
        )

    # 3) best по val_loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(
            model, optimizer, epoch + 1, global_step, config,
            ckpt_dir / "best", best_val_loss
        )
        print(f"  ★ new best val_loss: {best_val_loss:.4f}")


# ─── Финал ───
save_checkpoint(
    model, optimizer, num_epochs, global_step, config,
    ckpt_dir / "final", best_val_loss
)