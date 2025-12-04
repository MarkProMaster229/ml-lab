import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from Tokenizer import TokenizerMy
class TransformerBlock(nn.Module):
    def __init__(self, sizeVector=256, num_heads=8):
        super().__init__()
        self.sizeVector = sizeVector
        #вынеси
        #self.Vectorization = nn.Embedding(vocabSize, sizeVector)
        #вынеси
        #self.posEmbed = nn.Embedding(maxLong, sizeVector)
        self.ln1 = nn.LayerNorm(sizeVector)
        self.attn = nn.MultiheadAttention(sizeVector, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(sizeVector)

        self.ff = nn.Sequential(
            nn.Linear(sizeVector, sizeVector*4),
            nn.GELU(),
            nn.Linear(sizeVector*4, sizeVector)
        )
        #вынеси
        #self.lm_head = nn.Linear(sizeVector, vocabSize)
    def forward(self, x, attn_mask=None):
        # Attention
        h = self.ln1(x)
        z = self.attn(h, h, h, attn_mask=attn_mask)[0]
        x = x + z  # residual

        # FeedForward
        h = self.ln2(x)
        z = self.ff(h)
        x = x + z  # residual

        return x
class TransformerMy(nn.Module):
    def __init__(self, vocabSize=120000,maxLong=100,sizeVector=256, num_layers=1):
        super().__init__()
        self.Vectorization = nn.Embedding(vocabSize, sizeVector)
        self.posEmbed = nn.Embedding(maxLong, sizeVector)
        self.layers = nn.ModuleList([
            TransformerBlock(sizeVector=sizeVector, num_heads=8)
            for _ in range(num_layers)
            ])
        
        self.lm_head = nn.Linear(sizeVector, vocabSize)
        

    def forward(self, x):
        B, T = x.shape
        
        tok = self.Vectorization(x)
        pos = self.posEmbed(torch.arange(T, device=x.device))
        pos = pos.unsqueeze(0).repeat(B, 1, 1)

        h = tok + pos
        attn_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()


        for layer in self.layers:
            h = layer(h, attn_mask=attn_mask)


        return self.lm_head(h)
    
#class BigTransformer(nn.Module):
#    def __init__(self, vocabSize=1000, sizeVector=256, num_layers=6, n_models=10):
#        super().__init__()
#        self.model = TransformerMy(vocabSize=vocabSize,sizeVector=sizeVector,num_layers=num_layers)
#        self.blocks = nn.ModuleList([TransformerMy(vocabSize=vocabSize, sizeVector=sizeVector, num_layers=num_layers) 
#                                     for _ in range(n_models)])

    #def forward(self, x):
    #    h = x
    #    for block in self.layers:#вот тут тоже странно! уточни! 
    #        h = block(h)
    #    return h
class WorkModel():
    
    def __init__(self,vocabSize=120000, sizeVector=256, num_layers=6, n_models=10):
        self.tokenizator = TokenizerMy()
        real_vocab_size = self.tokenizator.get_vocab_size() 
        self.vocabSize = real_vocab_size
        self.sizeVector = sizeVector
        self.numLayers = num_layers
        self.nModels = n_models
        self.maxLong = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.model = BigTransformer(
        #    vocabSize=vocabSize,
        #    sizeVector=sizeVector,
        #    num_layers=num_layers,
        #    n_models=n_models
        #).to(self.device)
        self.model = TransformerMy(
            vocabSize=self.vocabSize,
            maxLong=self.maxLong,
            sizeVector=sizeVector,
            num_layers=num_layers
        ).to(self.device)

        # или через self.tokenizator.tokenizer.vocab_size
        
        print(f"Vocab size в модели: {vocabSize}")
        print(f"Real vocab size: {real_vocab_size}")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        self.critericon = nn.CrossEntropyLoss()
    
    def include(self):
        tokenizator = TokenizerMy()
        datalouder = tokenizator.datalouder()

        for batch in datalouder:
            inputIds = batch['input_ids']
            labels = batch['labels']
        
        num_epochs = 3
        

        for epoch in range(num_epochs):
            print("эпоха{epox}")

            self.model.train()

            for batchINDX, batch in enumerate(datalouder):
                inputINDX = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputINDX)
                #верно ли я понимаю мы идем как бы назад но так то вперед потому что итеррируемся мы назад 
                #типо предсказать следующий токен потому что мы как бы шли в лево и -1 чтоб сделать шаг в право хз 
                shakeRight = outputs[:, :-1, :].contiguous()
                #ну очевидно что первый токен мы типо не предсказываем хз
                DONTtOUCHlEFT = labels[:, 1:].contiguous() 

                loss = self.critericon(
                    shakeRight.view(-1, shakeRight.size(-1)),#встряхнуть
                    DONTtOUCHlEFT.view(-1)
                )
                loss.backward()
                self.optimizer.step()
                if batchINDX % 10 == 0:  # каждые 10 батчей
                    print(f"  Batch {batchINDX}/{len(datalouder)} - Loss: {loss.item():.4f}")

start = WorkModel()
start.include()