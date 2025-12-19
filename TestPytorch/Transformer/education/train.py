from Mytransformers import TransformerRun
import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "/home/chelovek/Музыка/epoch_2"

class Train():
    def __init__(self):
        self.text = None
        self._model = None
        self.optimizer = None
        self._config = None
        self.tokinize = None
        self.tokinize = AutoTokenizer.from_pretrained(path)
        if self.tokinize.pad_token is None:
            self.tokinize.pad_token = self.tokinize.eos_token
    def loadWeight(self):
        if self._model is None:
            if self._config is None:
                self._config = torch.load(f"{path}/config.pth")
                self._model = TransformerRun(
                    vocabSize=self._config["vocabSize"],
                    maxLong=self._config["maxLong"], 
                    sizeVector=self._config["sizeVector"],
                    block=self._config["numLayers"]
                ).to(device)
            
            self._model.load_state_dict(
                torch.load(f"{path}/model_weights.pth", map_location=device),
            )
        return self._model
    
    def Optimizator(self):
        if self.optimizer == None:
            self.optimizer = optim.Adam(
                self._model.parameters(), 
                lr=self._config.get("lr", 2e-4)
            )
        return self.optimizer
    

    def tokinizer(self,input):
        return self.tokinize(
            input["text"],
            truncation=True,
            padding="max_length",
            max_length=self._config["maxLong"],
            return_attention_mask=True
        )
    
    def dataset(self, data_path):
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        dataset = load_dataset("json", data_files=data_path)
        train_dataset = dataset["train"]
        texts = []
        for example in train_dataset:
            full_text = f"{example['input']} {example['target']}"
            texts.append(full_text)
        print(f"полных текстов{len(texts)}")
        
        encoded = self.tokinize(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self._config["maxLong"],
            return_tensors="pt",
            return_attention_mask=True
            )
        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
                
            def __len__(self):
                return len(self.encodings["input_ids"])
            
            def __getitem__(self, idx):
                
                return {
                    "input_ids": self.encodings["input_ids"][idx],
                    "attention_mask": self.encodings["attention_mask"][idx]
                    }
                
        print(f"Примеров: {len(encoded['input_ids'])}")
        return TextDataset(encoded)

    
    def create_dataloader(self, dataset, batch_size=10):
        def collate_fn(batch):
            input_ids = torch.stack([item["input_ids"] for item in batch])
            attention_mask = torch.stack([item["attention_mask"] for item in batch])
            return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
            }
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
            )

#---------------------------------------------------------------
trainer = Train()
model = trainer.loadWeight()
optimizer = trainer.Optimizator()
config = trainer._config
tokenizer = trainer.tokinize
dataset = trainer.dataset("/home/chelovek/Музыка/finaly.json")
dataloader = trainer.create_dataloader(dataset, batch_size=11)

loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

epochs = 160
save_every = 1
save_path = "/home/chelovek/exper/newtrainedModel"
os.makedirs(save_path, exist_ok=True)

num_batches = len(dataloader)
print(f"Количество батчей за эпоху: {num_batches}")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        optimizer.zero_grad()

        logits = model(input_ids)
        
        logits = logits[:, :-1, :].contiguous()
        targets = input_ids[:, 1:].contiguous()
        
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / num_batches
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
#разве тут что-то, когда-то было? Впрочем, это не имеет значение. Ведь так ? 