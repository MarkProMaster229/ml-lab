import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
# ----------------------------
# 1. Определение модели
# ----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, sizeVector=64, numHeads=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(sizeVector)
        self.attn = nn.MultiheadAttention(sizeVector, numHeads, batch_first=True)
        self.ln2 = nn.LayerNorm(sizeVector)
        self.ff = nn.Sequential(
            nn.Linear(sizeVector, sizeVector*4),
            nn.GELU(),
            nn.Linear(sizeVector*4, sizeVector),
        )

    def forward(self, x, attMask=None):
        h = self.ln1(x)
        z, _ = self.attn(h, h, h, attn_mask=attMask)
        h = self.ln2(x)
        z1 = self.ff(h)
        return x + z1

class TransformerRun(nn.Module):
    def __init__(self, vocabSize=120000, maxLong=200, sizeVector=64, block=2):
        super().__init__()
        self.maxLong = maxLong
        self.tokenEmbed = nn.Embedding(vocabSize, sizeVector)
        self.posEmbed = nn.Embedding(maxLong, sizeVector)
        self.ln_f = nn.LayerNorm(sizeVector)
        self.layers = nn.ModuleList([TransformerBlock(sizeVector=sizeVector) for _ in range(block)])
        self.lmHead = nn.Linear(sizeVector, vocabSize)

    def forward(self, x):
        B, T = x.shape
        tok = self.tokenEmbed(x)
        pos = self.posEmbed(torch.arange(T, device=x.device)).unsqueeze(0).repeat(B, 1, 1)
        h = tok + pos
        attMask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        for layer in self.layers:
            h = layer(h, attMask=attMask)
        h = self.ln_f(h)
        return self.lmHead(h)

# ----------------------------
# 2. Загружаем модель, токенизатор, оптимизатор
# ----------------------------
path = "/home/chelovek/exper/trained_model60"
config = torch.load(f"{path}/config.pth")
tokenizer = AutoTokenizer.from_pretrained(path)

model = TransformerRun(
    vocabSize=config['vocabSize'],
    maxLong=config['maxLong'],
    sizeVector=config['sizeVector'],
    block=config['numLayers']
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load(f"{path}/model_weights.pth", map_location=device))

optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 1e-4))
optimizer.load_state_dict(torch.load(f"{path}/optimizer.pth", map_location=device))

model.train()

# ----------------------------
# 3. Подготовка датасета с паддингом
# ----------------------------
dataset_path = "/home/chelovek/gatasetexp/simple.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    new_dataset = [line.strip() for line in f if line.strip()]

class MyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.data = [tokenizer.encode(t, truncation=True, max_length=max_length) for t in texts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

def collate_fn(batch):
    # Выравниваем последовательности по длине батча
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)

dataset = MyDataset(new_dataset, tokenizer, config['maxLong'])
dataloader = DataLoader(dataset, batch_size=12, shuffle=True, collate_fn=collate_fn)
# ----------------------------
# 4. Цикл дообучения с логами по эпохам и ReduceLROnPlateau
# ----------------------------
loss_fn = nn.CrossEntropyLoss()
epochs = 10       # общее количество новых эпох
save_every = 5    # сохранять каждые N эпох
save_path = "/home/chelovek/exper/newtrainedModel"
os.makedirs(save_path, exist_ok=True)

# Настроим scheduler
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min',       # уменьшаем LR, когда loss не падает
    factor=0.5,       # LR будет уменьшаться в 2 раза
    patience=2,       # ждать 2 эпохи без улучшения
    verbose=True
)

num_batches = len(dataloader)
print(f"Количество батчей за эпоху: {num_batches}")

for epoch in range(epochs):
    epoch_loss = 0.0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits.view(-1, logits.size(-1)), batch.view(-1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    # Средний loss за эпоху
    avg_epoch_loss = epoch_loss / num_batches
    
    # Обновляем LR по ReduceLROnPlateau
    scheduler.step(avg_epoch_loss)
    
    # Выводим лог: эпоха, средний loss и текущий LR
    current_lr = optimizer.param_groups[0]['lr']
    print(f"[Эпоха {epoch+1}/{epochs}] Avg Loss: {avg_epoch_loss:.4f}, LR: {current_lr:.6f}")

    # Сохраняем каждые 5 эпох
    if (epoch + 1) % save_every == 0:
        epoch_save_path = os.path.join(save_path, f"epoch_{epoch+1}")
        os.makedirs(epoch_save_path, exist_ok=True)
        torch.save(model.state_dict(), f"{epoch_save_path}/model_weights.pth")
        torch.save(optimizer.state_dict(), f"{epoch_save_path}/optimizer.pth")
        torch.save(config, f"{epoch_save_path}/config.pth")
        tokenizer.save_pretrained(epoch_save_path)
        print(f"💾 Модель после {epoch+1} эпох сохранена в {epoch_save_path}")

# ----------------------------
# 5. Сохраняем финальную модель
# ----------------------------
torch.save(model.state_dict(), f"{path}/model_weights.pth")
torch.save(optimizer.state_dict(), f"{path}/optimizer.pth")
torch.save(config, f"{path}/config.pth")
tokenizer.save_pretrained(path)
print(f"✅ Финальная модель сохранена в {path}")
