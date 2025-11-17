#данное архитектурно решение - Accuracy: 25/26 = 96.15%
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms
#delete
import os
SAVE_DIR = "saved_images"
os.makedirs(SAVE_DIR, exist_ok=True)
#delete
torch.set_num_threads(20)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# загружаем тренировочный и тестовый наборы
#train_dataset = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
#test_dataset = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

#теперь беру другой
ds = load_dataset("Leeps/Fonts-Individual-Letters", split="train")

alphabet = [chr(i) for i in range(65, 91)]

split_ds = ds.train_test_split(test_size=0.2, seed=42)

train_dataset = split_ds['train']
test_dataset = split_ds['test']

from PIL import Image
import numpy as np
import re
alphabet = [chr(i) for i in range(65, 91)]

from PIL import Image, ImageOps

def transform_fn(example):
    # Получаем картинку
    img = example["image"]
    
    # Если img хранится как список → превращаем в массив
    if isinstance(img, list):
        img = np.array(img, dtype=np.uint8)
        while img.ndim > 2:  # убираем лишние размерности
            img = img[0]
        img = Image.fromarray(img)
    
    # Переводим в одноканальный режим
    img = img.convert("L")
    
    # Если фон тёмный, буквы светлые → инвертируем
    # Берём среднее значение: если меньше 128 → фон тёмный
    if np.mean(np.array(img)) < 128:
        img = ImageOps.invert(img)
    
    # Применяем стандартные трансформации
    example["image"] = transform(img)
    
    # Берём текст из датасета
    text_list = example["text"]
    if isinstance(text_list, list):
        text = text_list[0]
    else:
        text = text_list
    
    # Парсим символ
    match = re.search(r"TOK '(.?)'", text)
    if match:
        symbol = match.group(1).upper()
        if symbol in alphabet:
            example["label"] = alphabet.index(symbol)
        else:
            return None  # игнорируем символы, которых нет в алфавите
    else:
        return None  # игнорируем, если не распарсилось
    
    return example


from torch.utils.data import DataLoader

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    images = torch.stack([b['image'] for b in batch])
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    return {'image': images, 'label': labels}

train_dataset = [ex for ex in train_dataset if transform_fn(ex) is not None]
test_dataset = [ex for ex in test_dataset if transform_fn(ex) is not None]

print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # сверточный слой номер 1 - 1 канал входа → 16 фильтров
        #1 канал входа у нас чёрно-белая картинка (1 канал).
        #16 фильтров (выход) → сеть создаёт 16 «карта признаков» 
        #(feature maps), каждая пытается выделить разные шаблоны, например линии или углы.
        #kernel_size=3 → каждый фильтр «смотрит» на маленький квадрат 3×3
        #padding=1 пустые пиксели по краям, чтобы размер картинки не уменьшался после свёртки.
        self.conv1 = nn.Conv2d(1,64,kernel_size=3,padding=1)
        
        #Делает уменьшение картинки вдвое
        #для того чтоб второй слой ловил больше паттернов
        #forward x = F.max_pool2d(x, 2)
        
        # Второй сверточный слой: 16 → 32 фильтра
        #Берёт 16 входных «карт признаков» с предыдущего слоя → создаёт 32 новые карты признаков
        #сеть - может комбинировать простые паттерны в сложные формы
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        #БАМ! не ожидали?
        #self.pool = nn.AdaptiveAvgPool2d((8,8))
        
        #Flatten + Linear — полносвязный слой
        #view превращает 3D-тензор (32×7×7) в 1D вектор
        #forward x = x.view(x.size(0), -1)
        #Linear — обычный нейронный слой: каждый вход соединяется со всеми 64 нейронами.
        self.fc1 = nn.Linear(1024*8*8, 512)
        #self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        #выходной слой 
        self.fc4 = nn.Linear(128, 26)
        
    def forward(self, x):
        #делает 16 карт признаков:
        x = F.relu(self.conv1(x))
        #Делит картинку на 2
        x = F.max_pool2d(x,2)
        #сеть получает уменьшенную картинку
        x = F.relu(self.conv2(x))
        #снова размер делится на 2
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv4(x))
        
        x = F.relu(self.conv5(x))
        #воооооот!
        #x = self.pool(x)
        
        #print(x.shape)
        
        #Выравнивание
        x = x.view(x.size(0), -1)
        #далее подать в полносвязный слой
        x = F.relu(self.fc1(x))
        #x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
            
        #выходной слой 
        x = self.fc4(x)
        return x
        
    #даталоудеры
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=160, shuffle=True, collate_fn=collate_fn
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=160, shuffle=False, collate_fn=collate_fn
)
train_losses = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)#???

for epoch in range(35):
    running_loss = 0
    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    
from safetensors.torch import save_file
#save_file(model.state_dict(), "cnn_letters.safetensors")
save_file(model.state_dict(), "/content/drive/MyDrive/cnn_letters.safetensors")

import matplotlib.pyplot as plt
plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.grid(True)
plt.savefig("training_loss15UP.png")


#dowload my parametrs(models)
#from safetensors.torch import load_file

#weights_dict = load_file("cnn_letters.safetensors")
#model = CNN()  # create new model
#model.load_state_dict(weights_dict)  # upload parametrs