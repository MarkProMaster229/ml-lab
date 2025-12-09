import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau



#Transformer Block

class TransformerBlock(nn.Module):
    def __init__(self, sizeVector=128, numHeads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(sizeVector)
        self.attn = nn.MultiheadAttention(sizeVector, numHeads, batch_first=True)
        self.ln2 = nn.LayerNorm(sizeVector)
        self.ff = nn.Sequential(
            nn.Linear(sizeVector, sizeVector * 4),
            nn.GELU(),
            nn.Linear(sizeVector * 4, sizeVector),
        )

    def forward(self, x, attMask=None):
        h = self.ln1(x)
        z, _ = self.attn(h, h, h, attn_mask=attMask)
        x = x + z

        h = self.ln2(x)
        z = self.ff(h)
        x = x + z

        return x


#Transformer Run

class TransformerRun(nn.Module):
    def __init__(self, vocabSize=120000, maxLong=256, sizeVector=128, block=4):
        super().__init__()
        self.maxLong = maxLong
        self.tokenEmbed = nn.Embedding(vocabSize, sizeVector)
        self.posEmbed = nn.Embedding(maxLong, sizeVector)
        self.ln_f = nn.LayerNorm(sizeVector)

        self.layers = nn.ModuleList([
            TransformerBlock(sizeVector=sizeVector, numHeads=4)
            for _ in range(block)
        ])

        self.lmHead = nn.Linear(sizeVector, vocabSize)

    def forward(self, x):
        B, T = x.shape
        tok = self.tokenEmbed(x)
        pos = self.posEmbed(torch.arange(T, device=x.device)).unsqueeze(0)
        h = tok + pos

        #Правильная causal mask
        # causal mask была сломана !!!!!!!
        #causal maskcausal maskcausal maskcausal maskcausal maskcausal maskcausal mask
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        attMask = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device),
            diagonal=1
        )

        for layer in self.layers:
            h = layer(h, attMask)

        h = self.ln_f(h)
        return self.lmHead(h)


path = "/home/chelovek/Музыка/epoch_10"
config = torch.load(f"{path}/config.pth")

tokenizer = AutoTokenizer.from_pretrained(path)

model = TransformerRun(
    vocabSize=config["vocabSize"],
    maxLong=config["maxLong"],
    sizeVector=config["sizeVector"],
    block=config["numLayers"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.load_state_dict(torch.load(f"{path}/model_weights.pth", map_location=device))

optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 1e-4))
optimizer.load_state_dict(torch.load(f"{path}/optimizer.pth", map_location=device))

model.train()



dataset_path = "/home/chelovek/Загрузки/MydatasetT.json"

with open(dataset_path, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]


class MyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.data = [
            tokenizer.encode(t, truncation=True, max_length=max_length)
            for t in texts
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)


def collate_fn(batch):
    return torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=tokenizer.pad_token_id
    )


dataset = MyDataset(texts, tokenizer, config["maxLong"])
dataloader = DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=collate_fn)


loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

epochs = 30
save_every = 5

save_path = "/home/chelovek/exper/newtrainedModel"
os.makedirs(save_path, exist_ok=True)

scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=1, verbose=True
)

num_batches = len(dataloader)
print(f"Количество батчей за эпоху: {num_batches}")

for epoch in range(epochs):
    epoch_loss = 0.0

    for batch in dataloader:
        batch = batch.to(device)

        optimizer.zero_grad()

        logits = model(batch)
        logits = logits[:, :-1, :].contiguous()
        targets = batch[:, 1:].contiguous()
        
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / num_batches
    scheduler.step(avg_loss)

    print(f"[Эпоха {epoch+1}/{epochs}] Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    if (epoch + 1) % save_every == 0:
        epoch_dir = os.path.join(save_path, f"epoch_{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)

        torch.save(model.state_dict(), f"{epoch_dir}/model_weights.pth")
        torch.save(optimizer.state_dict(), f"{epoch_dir}/optimizer.pth")
        torch.save(config, f"{epoch_dir}/config.pth")
        tokenizer.save_pretrained(epoch_dir)

        print(f"Модель сохранена: {epoch_dir}")


torch.save(model.state_dict(), f"{path}/model_weights.pth")
torch.save(optimizer.state_dict(), f"{path}/optimizer.pth")
torch.save(config, f"{path}/config.pth")
tokenizer.save_pretrained(path)

print(f"Финальная модель сохранена в {path}")
