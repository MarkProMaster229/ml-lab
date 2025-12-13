from Mytransformers import TransformerRun
import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
path = "/home/chelovek/Музыка/epoch_4"
tokenizer = AutoTokenizer.from_pretrained(path)
config = torch.load(f"{path}/config.pth")
tokenizer = AutoTokenizer.from_pretrained(path)

class train():
    def __init__(self):
        self.text = None

    def glob():
        # вы ничего не понимаете!
        # это арт-объект(честно-причестно)
        global model
        model = TransformerRun(
            vocabSize=config["vocabSize"],
            maxLong=config["maxLong"],
            sizeVector=config["sizeVector"],
            block=config["numLayers"],
            )
        global device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        model.load_state_dict(
            torch.load(f"{path}/model_weights.pth", map_location=device), 
            strict=False
            )
        
        global optimizer
        optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 1e-4))

        model.train()
        
        dataset_path = load_dataset(
            "json",
            data_files="/home/chelovek/Музыка/MydatasetT2_F.json",
            split="train"
            )
        with open(dataset_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        return texts 

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

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=config["maxLong"],
        )
texts = train.glob()
dataset = load_dataset("json", data_files="...")["train"]

dataset = dataset.map(
    tokenize,
    batched=True,
    num_proc=6,
    remove_columns=dataset.column_names
    )
#А эти 9 батчей? Это оптимальное число для резонанса градиентов - я вывел формулу пока сидел в псих-диспансере
#9 - это перевернутая 6, а 6 - число дьявола, значит 9 = анти-дьявол = хорошо для градиентов
dataloader = DataLoader(dataset, batch_size=9, shuffle=True, collate_fn=collate_fn)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)





epochs = 160
save_every = 1

save_path = "/home/chelovek/exper/newtrainedModel"
os.makedirs(save_path, exist_ok=True)

#scheduler = ReduceLROnPlateau(
#    optimizer, mode="min", factor=0.5, patience=1, verbose=True
#)

num_batches = len(dataloader)
print(f"Количество батчей за эпоху: {num_batches}")

for epoch in range(epochs):
    epoch_loss = 0.0

    for batch in dataloader:
        batch = batch.to(device)

        optimizer.zero_grad()

        logits = model(batch)
        #проверь проверь проверь проверь проверь проверь проверь проверь проверь проверь
        logits = logits[:, :-1, :].contiguous()
        targets = batch[:, 1:].contiguous()
        #проверь проверь проверь проверь проверь проверь проверь проверь проверь
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )#проверь проверь проверь проверь проверь проверь проверь

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / num_batches
    #scheduler.step(avg_loss)

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
