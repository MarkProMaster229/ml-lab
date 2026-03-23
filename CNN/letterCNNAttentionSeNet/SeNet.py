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


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation блок.
    reduction_ratio – коэффициент сжатия каналов (обычно 16).
    """
    def __init__(self, channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        reduced_channels = max(1, channels // reduction_ratio)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # Squeeze: глобальный пулинг
            nn.Conv2d(channels, reduced_channels, kernel_size=1), # Excitation: FC сжатие
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1),  # Excitation: расширение
            nn.Sigmoid()                       # Выход – веса важности каналов
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale                        # Масштабирование исходных признаков

class SEBasicBlock(nn.Module):
    """
    BasicBlock для ResNet18 с SE-блоком после суммирования (как в оригинале).
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, reduction_ratio=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (projection if needed)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # SE блок после сложения
        self.se = SEBlock(out_channels, reduction_ratio)

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        out += identity               # Residual connection
        out = self.se(out)             # Squeeze-and-Excitation
        out = F.relu(out, inplace=True)

        return out

class SE_ResNet18(nn.Module):
    """
    Полная модель SE-ResNet18 с 1 входным каналом.
    """
    def __init__(self, num_classes=64, reduction_ratio=16):
        super(SE_ResNet18, self).__init__()

        # Начальный слой
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Четыре residual stage с SE-блоками
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1, reduction_ratio=reduction_ratio)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2, reduction_ratio=reduction_ratio)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2, reduction_ratio=reduction_ratio)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2, reduction_ratio=reduction_ratio)

        # Классификатор
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride, reduction_ratio):
        layers = []
        layers.append(SEBasicBlock(in_channels, out_channels, stride, reduction_ratio))
        for _ in range(1, blocks):
            layers.append(SEBasicBlock(out_channels, out_channels, stride=1, reduction_ratio=reduction_ratio))
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
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
train_losses = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = SE_ResNet18().to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Всего параметров: {total_params:,}")


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
    torch.save(model.state_dict(), f'cnn_model_epoch_{epoch+1}.pth')
    print(f"Модель сохранена: cnn_model_epoch_{epoch+1}.pth")

#Epoch  1 | Train Loss: 1.4508 | Val Loss: 0.5244 | Val Acc: 85.04%
#Модель сохранена: cnn_model_epoch_1.pth
#Epoch  2 | Train Loss: 0.4297 | Val Loss: 0.3719 | Val Acc: 88.88%
#Модель сохранена: cnn_model_epoch_2.pth
#Epoch  3 | Train Loss: 0.3424 | Val Loss: 0.3682 | Val Acc: 89.00%
#Модель сохранена: cnn_model_epoch_3.pth
#Epoch  4 | Train Loss: 0.2957 | Val Loss: 0.3323 | Val Acc: 89.97%
#Модель сохранена: cnn_model_epoch_4.pth
#Epoch  5 | Train Loss: 0.2599 | Val Loss: 0.2791 | Val Acc: 91.13%
#Модель сохранена: cnn_model_epoch_5.pth
#Epoch  6 | Train Loss: 0.2317 | Val Loss: 0.2870 | Val Acc: 91.49%
#Модель сохранена: cnn_model_epoch_6.pth
#Epoch  7 | Train Loss: 0.2105 | Val Loss: 0.3122 | Val Acc: 90.82%
#Модель сохранена: cnn_model_epoch_7.pth
#Epoch  8 | Train Loss: 0.1951 | Val Loss: 0.2713 | Val Acc: 92.02%
#Модель сохранена: cnn_model_epoch_8.pth
#Epoch  9 | Train Loss: 0.1782 | Val Loss: 0.2380 | Val Acc: 92.72%
#Модель сохранена: cnn_model_epoch_9.pth
#Epoch 10 | Train Loss: 0.1636 | Val Loss: 0.2543 | Val Acc: 92.24%
#Модель сохранена: cnn_model_epoch_10.pth
#Epoch 11 | Train Loss: 0.1533 | Val Loss: 0.2500 | Val Acc: 92.41%
#Модель сохранена: cnn_model_epoch_11.pth
#Epoch 12 | Train Loss: 0.1434 | Val Loss: 0.2589 | Val Acc: 92.41%
#Модель сохранена: cnn_model_epoch_12.pth
#Epoch 13 | Train Loss: 0.1370 | Val Loss: 0.2367 | Val Acc: 92.91%
#Модель сохранена: cnn_model_epoch_13.pth
#Epoch 14 | Train Loss: 0.1272 | Val Loss: 0.2691 | Val Acc: 92.51%
#Модель сохранена: cnn_model_epoch_14.pth
#Epoch 15 | Train Loss: 0.1192 | Val Loss: 0.2576 | Val Acc: 92.52%
#Модель сохранена: cnn_model_epoch_15.pth
#Epoch 16 | Train Loss: 0.1127 | Val Loss: 0.2795 | Val Acc: 92.05%
#Модель сохранена: cnn_model_epoch_16.pth
#Epoch 17 | Train Loss: 0.1054 | Val Loss: 0.2685 | Val Acc: 92.43%
#Модель сохранена: cnn_model_epoch_17.pth
#Epoch 18 | Train Loss: 0.1002 | Val Loss: 0.2932 | Val Acc: 91.41%
#Модель сохранена: cnn_model_epoch_18.pth
#Epoch 19 | Train Loss: 0.0922 | Val Loss: 0.2923 | Val Acc: 92.19%
#Модель сохранена: cnn_model_epoch_19.pth
#Epoch 20 | Train Loss: 0.0908 | Val Loss: 0.2934 | Val Acc: 92.11%
#Модель сохранена: cnn_model_epoch_20.pth