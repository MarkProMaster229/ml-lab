import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
class TransformerMy(nn.Module):
    def __init__(self, vocabSize=1000, sizeVector=256, maxLong=100):
        super().__init__()
        self.sizeVector = sizeVector
        
        self.Vectorization = nn.Embedding(vocabSize, sizeVector)
        self.posEmbed = nn.Embedding(maxLong, sizeVector)
        
        self.ln1 = nn.LayerNorm(sizeVector)
        self.attn = nn.MultiheadAttention(sizeVector, 8, batch_first=True)
        self.ln2 = nn.LayerNorm(sizeVector)
        
        self.ff = nn.Sequential(
            nn.Linear(sizeVector, sizeVector*4),
            nn.GELU(),
            nn.Linear(sizeVector*4, sizeVector)
        )
        self.lm_head = nn.Linear(sizeVector, vocabSize)
        self.register_buffer("attn_mask", None)
    #эта функция нужна только для меня

    #def inpute(self):
        #batchSize = 2
        #seqLen = 15
        #vocabSize = 1000
        #input_ids = torch.randint(0, vocabSize, (batchSize, seqLen ))
        
        #полезно благодаря нему превращаем индексы в ветора 
        #пусть размерность вектора будет 256
        #sizeVector = 256
        
        #Vectorization = torch.nn.Embedding(vocabSize,sizeVector)
        #x = Vectorization(input_ids)
        
        #maxLong = 100
        #posEmbed = torch.nn.Embedding(maxLong,sizeVector)
        #position = torch.arange(seqLen).unsqueeze(0)
        #x = x + posEmbed(position)
        
        #mask = torch.triu(torch.ones(seqLen, seqLen) * float('-inf'), diagonal=1)
        
    def create_mask(self, seq_len):
        return torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        
    def forward(self, input_ids):
        #размер батча в последовательности нужен нам далее
        batch_Size, seq_Len = input_ids.shape
        device = input_ids.device
        #далее создать маску внимания,я буду пользоваться - верхнетреугольной маской
        #создается маска так
        if self.attn_mask is None or self.attn_mask.size(0) != seq_Len:
            self.attn_mask = self.create_mask(seq_Len).to(device)
        #далее преобразоватьс индексы токенов в эмбединги 
        # self.Vectorization - это слой nn.Embedding, который обучается
        # token_embeds имеет форму (batch_size, seq_len, sizeVector)
        token_embeds = self.Vectorization(input_ids)
        #далее позиционные эмбединги под инициализацию
        #но в начале тензор позиций 
        position = torch.arange(seq_Len, device=device).unsqueeze(0)# форма (1, seq_len)
        # Получаем позиционные эмбеддинги для каждой позиции
        pos_embeds = self.posEmbed(position)  # форма (1, seq_len, sizeVector)
        #тут просто стандартное сложение сливаем позиционный тензор с основным тензором
        x = token_embeds + pos_embeds

        #далее самое интересное теперь реализуем блоки трансформера Self-Attention с остаточным соединением
        #зачем ? ну как я понял чтоб в больших сетях(не наш случай сейчас само собой) градиенты не затухали
        #так называемые резидуальные связи
        residual = x
        # слой нормализации к основному тензору
        x = self.ln1(x)
        #как выглядит в pytorch механизм внимания ? а вот так - 
        # self.attn - это nn.MultiheadAttention
        # Мы передаем x как query, key и value (self-attention)
        # attn_mask - маска, чтобы токен не видел будущие токены
        # attn_output имеет форму (batch_size, seq_len, sizeVector)
        attn_output, _ = self.attn(x, x, x, attn_mask=self.attn_mask)
        #вот теперь то применим резидуальную связь то есть не нормализованый слой 
        #зачем мы сливаем грязный резидуальный слой(фактически основной тензор до нормализации ?)
        #это нужно для того чтоб если attention испортил все, грубо говоря потеряв исходную информацию 
        #резидуальная связь выступила последним рубежом
        # Градиент: dL/dx = dL/d(x+F) * ( 1 + dF/dx)
        #                      ↑          ↑
        # У нас появилась "1" в градиенте!
        #так как лично до меня доходит туго мне gpt нарисовал схему, мне лично куда понятнее 
        # Исходный X (грязный)
            #│
            #├───────┐
            #│       │
            #▼       │
          #ln1(x)    │
            #│       │
            #▼       │
          #attn(x)   │
            #│       │
            #▼       │
            #+ ◄─────┘
            #│
            #▼
      #Новый X (грязный + чистое внимание)

        #выходам из слоя внимания 
        x = residual + attn_output  # форма (batch_size, seq_len, sizeVector)
        # 6. Второй блок трансформера: Feed-Forward Network с остаточным соединением
        # Сохраняем вход блока (x) для остаточного соединения
        residual = x
        #Применяем LayerNorm
        x = self.ln2(x)
        # Применяем feed-forward network (два линейных слоя с активацией GELU между ними)
        ffn_output = self.ff(x)  # форма (batch_size, seq_len, sizeVector)
        # Добавляем резидуальную связь
        x = residual + ffn_output  # форма (batch_size, seq_len, sizeVector)

        #заканчиваем с вниманием и переходим к выходу 
        #проекция на словарь 
        #преобразуем вектор размерности sizeVector в вектор размерности vocabSize
        # Это делается для того, чтобы для каждой позиции в последовательности получить оценку (логит) для каждого токена в словаре
        logits = self.lm_head(x)  # форма (batch_size, seq_len, vocabSize)
        return logits
    
#создам чисто для примера
model = TransformerMy(vocabSize=1000, sizeVector=256, maxLong=100)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
def create_synthetic_batch(batch_size, seq_len, vocab_size, device):
    """Создает случайный батч данных для тестирования"""
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.roll(inputs, shifts=-1, dims=1)  # сдвигаем на 1 вправо
    targets[:, -1] = -100  # последний токен не имеет цели
    return inputs, targets

# Функция для вычисления потерь
def compute_loss(logits, targets, vocab_size):
    """Вычисляет кросс-энтропийную потерь"""
    return F.cross_entropy(
        logits.view(-1, vocab_size),# (batch*seq, vocab)
        targets.view(-1),# (batch*seq)
        ignore_index=-100# игнорируем padding/последние токены
    )
import time
BATCH_SIZE = 8
SEQ_LEN = 32
VOCAB_SIZE = 1000
STEPS_PER_EPOCH = 100
print(f"Начинаем обучение на устройстве: {device}")
print("-" * 50)

for epoch in range(10):
    epoch_start_time = time.time()
    total_loss = 0
    model.train()
    for step in range(STEPS_PER_EPOCH):
        # 2.1 Подготовка батча (в реальности это будет из DataLoader)
        inputs, targets = create_synthetic_batch(
            BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, device=device
        )
        # 2.2 Обнуляем градиенты с предыдущего шага
        optimizer.zero_grad()
        # 2.3 Прямой проход (forward pass)
        logits = model(inputs)  # вызывает метод forward
        # 2.4 Вычисление потерь
        loss = compute_loss(logits, targets, VOCAB_SIZE)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if step % 20 == 0:
            print(f"Эпоха {epoch+1}, Шаг {step}: Loss = {loss.item():.4f}")
    epoch_time = time.time() - epoch_start_time
    avg_loss = total_loss / STEPS_PER_EPOCH

    print("-" * 50)
    print(f"Эпоха {epoch+1} завершена за {epoch_time:.1f} сек")
    print(f"Средний Loss: {avg_loss:.4f}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print("-" * 50)
print("завершено")