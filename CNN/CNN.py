from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# загружаем тренировочный и тестовый наборы
train_dataset = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
test_dataset = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # сверточный слой номер 1 - 1 канал входа → 16 фильтров
        #1 канал входа у нас чёрно-белая картинка (1 канал).
        #16 фильтров (выход) → сеть создаёт 16 «карта признаков» 
        #(feature maps), каждая пытается выделить разные шаблоны, например линии или углы.
        #kernel_size=3 → каждый фильтр «смотрит» на маленький квадрат 3×3
        #padding=1 пустые пиксели по краям, чтобы размер картинки не уменьшался после свёртки.
        self.conv1 = nn.Conv2d(1,16,kernel_size=3,padding=1)
        
        #Делает уменьшение картинки вдвое
        #для того чтоб второй слой ловил больше паттернов
        #forward x = F.max_pool2d(x, 2)
        
        # Второй сверточный слой: 16 → 32 фильтра
        #Берёт 16 входных «карт признаков» с предыдущего слоя → создаёт 32 новые карты признаков
        #сеть - может комбинировать простые паттерны в сложные формы
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        #Flatten + Linear — полносвязный слой
        #view превращает 3D-тензор (32×7×7) в 1D вектор
        #forward x = x.view(x.size(0), -1)
        #Linear — обычный нейронный слой: каждый вход соединяется со всеми 64 нейронами.
        self.fc1 = nn.Linear(32*7*7, 64)
        
        #выходной слой 
        self.fc2 = nn.Linear(64, 26)
        
    def forward(self, x):
        #делает 16 карт признаков:
        x = F.relu(self.conv1(x))
        #Делит картинку на 2
        x = F.max_pool2d(x,2)
        #сеть получает уменьшенную картинку
        x = F.relu(self.conv2(x))
        #снова размер делится на 2
        x = F.max_pool2d(x, 2)
        
        #print(x.shape)
        
        #Выравнивание
        x = x.view(x.size(0), -1)
        #далее подать в полносвязный слой
        x = F.relu(self.fc1(x))
            
        #выходной слой 
        x = self.fc2(x)
        return x
        
    #даталоудеры
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=500, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=500, shuffle=False)
    
model = CNN()
    
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

train_losses = []


for epoch in range(30):
    running_loss = 0
    for images, labels in train_loader:
        labels = labels - 1
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        #print(model.conv1.weight.grad.shape)
        #print(model.conv1.weight.grad) 
        optimizer.step()
        running_loss += loss.item()


    
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
from safetensors.torch import save_file
save_file(model.state_dict(), "cnn_letters.safetensors")

import matplotlib.pyplot as plt
plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.grid(True)
plt.savefig("training_loss30.png")


#dowload my parametrs(models)
#from safetensors.torch import load_file

#weights_dict = load_file("cnn_letters.safetensors")
#model = CNN()  # create new model
#model.load_state_dict(weights_dict)  # upload parametrs