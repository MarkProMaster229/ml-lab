from decoderOnly import TransformerBlock
from decoderOnly import TransformerRun
import torch
import torch.optim as optim
import torch.nn as nn
from tokenizer import TokenizerMy
class WorkModel():
    def __init__(self, sizeVector = 512, num_layers=12, maxLong=100):
        self.tokenizator = TokenizerMy()
        self.vocabSize = self.tokenizator.get_vocab_size()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sizeVector = sizeVector
        self.num_layers = num_layers
        self.maxLong = maxLong

        self.model = TransformerRun(
            vocabSize=self.vocabSize,
            maxLong=self.maxLong,
            sizeVector=sizeVector,
            block=num_layers
        ).to(self.device)
        print(f"Размер словаря: {self.vocabSize}")
        print(f"Размер эмбеддинга: {sizeVector}")
        print(f"Количество блоков: {num_layers}")
        print(f"Устройство: {self.device}")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
    
    def workModel(self):
        tokenizator = self.tokenizator
        datalouder = tokenizator.datalouder()
        numEpoch = 1

        for epoch in range(numEpoch):
            print(f"эпоха{epoch}")
            self.model.train()

            for batchINDX, batch in enumerate(datalouder):
                inputINDX = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputINDX)
                shakeRight = outputs[:, :-1, :].contiguous()
                DONTtOUCHlEFT = labels[:, 1:].contiguous() 

                loss = self.criterion(
                    shakeRight.view(-1, shakeRight.size(-1)),#встряхнуть
                    DONTtOUCHlEFT.view(-1)
                )
                loss.backward()
                self.optimizer.step()
                if batchINDX % 10 == 0:
                    print(f"  Batch {batchINDX}/{len(datalouder)} - Loss: {loss.item():.4f}")
#------------------------------------------------------------------------------------------------
    def save_model(self, path="my_model"):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), f"{path}/model_weights.pth")
        config = {
            'vocabSize': self.vocabSize,
            'maxLong': self.maxLong,
            'sizeVector': self.sizeVector,
            'numLayers': self.num_layers,
            'device': str(self.device)
            }
        torch.save(config, f"{path}/config.pth")
        self.tokenizator.tokenizer.save_pretrained(path)
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pth")
        print(f"Модель сохранена в папку: {path}")
        print(f"   - Веса модели: {path}/model_weights.pth")
        print(f"   - Конфигурация: {path}/config.pth")
        print(f"   - Токенизатор: {path}/")
#---------------------------------------------------------------------------------------------------

start = WorkModel()
start.workModel()
start.save_model("trained_model")