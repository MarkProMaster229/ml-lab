from decoderOnly import TransformerBlock
from decoderOnly import TransformerRun
import torch
import torch.optim as optim
import torch.nn as nn
from tokenizer import TokenizerMy
class WorkModel():
    def __init__(self, sizeVector = 512, num_layers=12):
        self.tokenizator = TokenizerMy()
        self.vocabSize = self.tokenizator.get_vocab_size()
        self.maxLong = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        numEpoch = 10

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

start = WorkModel()
start.workModel()