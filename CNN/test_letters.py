import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from safetensors.torch import load_file
import string

# -----------------------------
# ТВОЯ МОДЕЛЬ 1в1
# -----------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
            
        #выходной слой 
        x = self.fc4(x)
        return x


model = CNN()
weights = load_file("cnn_letters.safetensors")
model.load_state_dict(weights)
model.eval()

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dir = "/home/chelovek/Изображения/words"
letters = list(string.ascii_uppercase)

correct = 0
total = 0

for letter in letters:
    path = os.path.join(test_dir, f"{letter}.png")

    if not os.path.exists(path):
        print(f"[WARN] Нет файла {letter}.png")
        continue

    img = Image.open(path).convert("L")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(x)
        pred_idx = output.argmax(dim=1).item()
        pred_letter = letters[pred_idx]

    total += 1
    if pred_letter == letter:
        correct += 1
        print(f"{letter}: OK  → {pred_letter}")
    else:
        print(f"{letter}: FAIL → {pred_letter}")

print("\n---------------------------")
print(f"Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
