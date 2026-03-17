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

class AttentionGate(nn.Module):
    """
    Механизм внимания, который позволяет глубоким слоям влиять на ранние.
    Принимает признаки из глубокого слоя и создает карту внимания для раннего слоя.
    """
    def __init__(self, deep_channels, early_channels, reduced_channels=16):
        super(AttentionGate, self).__init__()
        
        # Преобразуем глубокие признаки в карту внимания для ранних слоев
        self.deep_transform = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Глобальный контекст
            nn.Conv2d(deep_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, early_channels, kernel_size=1, bias=False),
            nn.Sigmoid()  # Карта внимания от 0 до 1
        )
        
        # Пространственное внимание (опционально, для более точной фокусировки)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(early_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, early_features, deep_features):
        # Создаем канальное внимание из глубоких признаков
        channel_attention = self.deep_transform(deep_features)
        
        # Применяем канальное внимание к ранним признакам
        attended_features = early_features * channel_attention
        
        # Добавляем пространственное внимание
        spatial_weights = self.spatial_attention(attended_features)
        attended_features = attended_features * spatial_weights
        
        return attended_features + early_features  # Residual connection


class ResNet18WithAttention(nn.Module):
    """
    ResNet18 с механизмом внимания, где глубокие слои подсказывают начальным.
    """
    def __init__(self, num_classes=64):
        super(ResNet18WithAttention, self).__init__()
        
        # Начальный слой
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Основные слои (как в классическом ResNet18)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)  # 56x56
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2) # 28x28
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2) # 14x14
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2) # 7x7
        
        # Механизмы внимания: каждый глубокий слой учит предыдущий
        self.attention_4_to_3 = AttentionGate(512, 256)  # layer4 -> layer3
        self.attention_3_to_2 = AttentionGate(256, 128)  # layer3 -> layer2
        self.attention_2_to_1 = AttentionGate(128, 64)   # layer2 -> layer1
        self.attention_1_to_conv = AttentionGate(64, 64) # layer1 -> conv1
        
        # Для сохранения признаков между слоями
        self.skip_connections = nn.ModuleList()
        
        # Pooling и классификатор
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        self._init_weights()
        
        # Для хранения промежуточных признаков
        self.intermediate_features = {}
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
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
        #ты не прав !
    def forward(self, x):
        # начальная свертка
        conv1 = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(conv1)

        # обычный ResNet проход
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # top-down attention
        layer3 = self.attention_4_to_3(layer3, layer4)
        layer2 = self.attention_3_to_2(layer2, layer3)
        layer1 = self.attention_2_to_1(layer1, layer2)
        conv1 = self.attention_1_to_conv(conv1, layer1)

        # классификация
        x = self.avgpool(layer4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    def get_attention_maps(self, x):
        """Метод для визуализации карт внимания"""
        self.forward(x)
        attention_maps = {}
        for name, features in self.intermediate_features.items():
            if 'attention' in name:
                attention_maps[name] = features.detach().cpu()
        return attention_maps
    
train_losses = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = ResNet18WithAttention().to(device)

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


#Epoch  1 | Train Loss: 1.1025 | Val Loss: 0.8558 | Val Acc: 76.52%
#Модель сохранена: cnn_model_epoch_1.pth
#Epoch  2 | Train Loss: 0.3954 | Val Loss: 0.3894 | Val Acc: 88.75%
#Модель сохранена: cnn_model_epoch_2.pth
#Epoch  3 | Train Loss: 0.3118 | Val Loss: 0.3378 | Val Acc: 89.72%
#Модель сохранена: cnn_model_epoch_3.pth
#Epoch  4 | Train Loss: 0.2583 | Val Loss: 0.2969 | Val Acc: 91.08%
#Модель сохранена: cnn_model_epoch_4.pth
#Epoch  5 | Train Loss: 0.2224 | Val Loss: 0.2771 | Val Acc: 91.72%
#Модель сохранена: cnn_model_epoch_5.pth
#Epoch  6 | Train Loss: 0.1945 | Val Loss: 0.2679 | Val Acc: 91.82%
#Модель сохранена: cnn_model_epoch_6.pth
#Epoch  7 | Train Loss: 0.1717 | Val Loss: 0.2703 | Val Acc: 91.85%
#Модель сохранена: cnn_model_epoch_7.pth
#Epoch  8 | Train Loss: 0.1545 | Val Loss: 0.2725 | Val Acc: 92.03%
#Модель сохранена: cnn_model_epoch_8.pth
#Epoch  9 | Train Loss: 0.1392 | Val Loss: 0.2705 | Val Acc: 92.28%
#Модель сохранена: cnn_model_epoch_9.pth
#Epoch 10 | Train Loss: 0.1268 | Val Loss: 0.2920 | Val Acc: 91.28%
#Модель сохранена: cnn_model_epoch_10.pth
#Epoch 11 | Train Loss: 0.1176 | Val Loss: 0.2681 | Val Acc: 92.42%
#Модель сохранена: cnn_model_epoch_11.pth
#Epoch 12 | Train Loss: 0.1068 | Val Loss: 0.2758 | Val Acc: 92.34%
#Модель сохранена: cnn_model_epoch_12.pth
#Epoch 13 | Train Loss: 0.0998 | Val Loss: 0.2757 | Val Acc: 92.65%
#Модель сохранена: cnn_model_epoch_13.pth
#Epoch 14 | Train Loss: 0.0949 | Val Loss: 0.2806 | Val Acc: 92.33%
#Модель сохранена: cnn_model_epoch_14.pth
#Epoch 15 | Train Loss: 0.0867 | Val Loss: 0.2718 | Val Acc: 92.53%
#Модель сохранена: cnn_model_epoch_15.pth