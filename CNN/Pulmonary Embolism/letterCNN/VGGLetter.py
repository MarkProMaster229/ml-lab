from Datalouder import FontLetterDataset, create_dataloaders
from torchvision import transforms

train_loader, test_loader, info = create_dataloaders(
    train_batch_size=32,
    test_batch_size=32,
    img_size=224,
    test_size=0.2
)

print(f"Train: {info['train_size']} примеров")
print(f"Test: {info['test_size']} примеров")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


import torch
import torch.nn as nn
import torch.nn.functional as F
#создали - gcc экплоит, systemd - вирус корпарация ужаса ред хед

#Эксперименты показали, что архитектура VGG16 без остаточных связей (skip-connections) 
#неспособна эффективно обучаться на относительно небольшом наборе данных 
#изображений букв (≈50 000 примеров). Из-за большой глубины сети градиенты при 
#обратном распространении затухают (vanishing gradient problem), и веса ранних слоёв 
#практически не обновляются. В результате модель сваливается в тривиальное решение — 
#предсказание равномерного распределения классов (loss ≈ ln(64) ≈ 4.15), что соответствует 
#accuracy на уровне случайного угадывания (≈1.5%).
class BasicBlock(nn.Module):
    """Базовый блок для ResNet18/34: два сверточных слоя 3x3 с BatchNorm и skip-connection"""
    expansion = 1  # для ResNet18 количество каналов не меняется внутри блока

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # Первый сверточный слой (может уменьшать размерность, если stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Второй сверточный слой (всегда stride=1, сохраняет размер)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip-connection: если размерность или количество каналов меняется,
        # используем свертку 1x1 для подгонки
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # Первый слой + ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # Второй слой (без ReLU перед сложением)
        out = self.bn2(self.conv2(out))
        # Добавляем skip-connection
        out += self.shortcut(x)
        # Итоговая активация
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    """
    ResNet18 для классификации grayscale изображений (1 канал)
    """
    def __init__(self, num_classes=64):
        super(ResNet18, self).__init__()
        
        # Начальный слой: большой фильтр 7x7, stride=2, padding=3
        # Вход: 1 канал (grayscale), выход: 64 карты признаков
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Четыре последовательных слоя (stage), каждый состоит из двух BasicBlock
        # В каждом следующем слое количество каналов удваивается, а размер уменьшается
        self.layer1 = self._make_layer(64, 64,  blocks=2, stride=1)  # 56x56
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2) # 28x28
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2) # 14x14
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2) # 7x7
        
        # В конце Adaptive Average Pool, который приводит каждый канал к 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Полносвязный классификатор (без скрытых слоёв, только выходной)
        self.fc = nn.Linear(512, num_classes)
        
        # Инициализация весов (Xavier/He — хороший тон)
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Создаёт последовательность из blocks блоков, начиная с возможно изменяющего размер"""
        layers = []
        # Первый блок может уменьшать размер (если stride=2)
        layers.append(BasicBlock(in_channels, out_channels, stride))
        # Остальные блоки сохраняют размер
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Начальная свертка + пулинг
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Четыре residual-слоя
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global Average Pooling
        x = self.avgpool(x)
        # Распрямляем в вектор
        x = x.view(x.size(0), -1)
        # Классификатор
        x = self.fc(x)
        return x
    
train_losses = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = ResNet18().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)#???


num_epochs = 25
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):

    model.train()
    running_loss = 0.0
    
    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)
    
    model.eval()
    val_running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_val_loss = val_running_loss / len(test_loader)
    epoch_val_acc = 100.0 * correct / total
    
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)
    
    print(f"Epoch {epoch+1:2d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")