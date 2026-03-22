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

import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_c, out_c, kernel, stride):
        padding = (kernel - 1) // 2
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):

    def __init__(self, in_c, out_c, stride, expand_ratio):
        super().__init__()

        hidden = int(in_c * expand_ratio)
        self.use_residual = stride == 1 and in_c == out_c

        layers = []

        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_c, hidden, 1, 1))

        layers += [
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),

            nn.Conv2d(hidden, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c)
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):

        if self.use_residual:
            return x + self.block(x)

        return self.block(x)


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=64, width_mult=1.0):
        super().__init__()

        def c(ch):
            return int(ch * width_mult)

        self.stem = ConvBNReLU(1, c(32), 3, 2)

        self.blocks = nn.Sequential(

            InvertedResidual(c(32), c(16), 1, 1),

            InvertedResidual(c(16), c(24), 2, 6),
            InvertedResidual(c(24), c(24), 1, 6),

            InvertedResidual(c(24), c(32), 2, 6),
            InvertedResidual(c(32), c(32), 1, 6),
            InvertedResidual(c(32), c(32), 1, 6),

            InvertedResidual(c(32), c(64), 2, 6),
            InvertedResidual(c(64), c(64), 1, 6),
            InvertedResidual(c(64), c(64), 1, 6),
            InvertedResidual(c(64), c(64), 1, 6),

            InvertedResidual(c(64), c(96), 1, 6),
            InvertedResidual(c(96), c(96), 1, 6),
            InvertedResidual(c(96), c(96), 1, 6),

            InvertedResidual(c(96), c(160), 2, 6),
            InvertedResidual(c(160), c(160), 1, 6),
            InvertedResidual(c(160), c(160), 1, 6),

            InvertedResidual(c(160), c(320), 1, 6)
        )

        self.last = ConvBNReLU(c(320), c(1280), 1, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(c(1280), num_classes)

    def forward(self, x):

        x = self.stem(x)

        x = self.blocks(x)

        x = self.last(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x
train_losses = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = MobileNetV2().to(device)

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



#Epoch  1 | Train Loss: 1.1325 | Val Loss: 0.5477 | Val Acc: 84.25%
#Модель сохранена: cnn_model_epoch_1.pth
#Epoch  2 | Train Loss: 0.4790 | Val Loss: 0.4733 | Val Acc: 86.94%
#Модель сохранена: cnn_model_epoch_2.pth
#Epoch  3 | Train Loss: 0.3903 | Val Loss: 0.3612 | Val Acc: 89.77%
#Модель сохранена: cnn_model_epoch_3.pth
#Epoch  4 | Train Loss: 0.3346 | Val Loss: 0.3245 | Val Acc: 90.35%
#Модель сохранена: cnn_model_epoch_4.pth
#Epoch  5 | Train Loss: 0.3021 | Val Loss: 0.3038 | Val Acc: 90.68%
#Модель сохранена: cnn_model_epoch_5.pth
#Epoch  6 | Train Loss: 0.2716 | Val Loss: 0.2841 | Val Acc: 91.16%
#Модель сохранена: cnn_model_epoch_6.pth
#Epoch  7 | Train Loss: 0.2520 | Val Loss: 0.3179 | Val Acc: 90.69%
#Модель сохранена: cnn_model_epoch_7.pth
#Epoch  8 | Train Loss: 0.2352 | Val Loss: 0.2777 | Val Acc: 91.56%
#Модель сохранена: cnn_model_epoch_8.pth
#Epoch  9 | Train Loss: 0.2168 | Val Loss: 0.2801 | Val Acc: 91.56%
#Модель сохранена: cnn_model_epoch_9.pth
#Epoch 10 | Train Loss: 0.2052 | Val Loss: 0.2672 | Val Acc: 91.91%
#Модель сохранена: cnn_model_epoch_10.pth
#Epoch 11 | Train Loss: 0.1914 | Val Loss: 0.2698 | Val Acc: 91.83%
#Модель сохранена: cnn_model_epoch_11.pth
#Epoch 12 | Train Loss: 0.1824 | Val Loss: 0.2501 | Val Acc: 92.17%
#Модель сохранена: cnn_model_epoch_12.pth
#Epoch 13 | Train Loss: 0.1713 | Val Loss: 0.2654 | Val Acc: 91.88%
#Модель сохранена: cnn_model_epoch_13.pth
#Epoch 14 | Train Loss: 0.1647 | Val Loss: 0.2573 | Val Acc: 92.28%
#Модель сохранена: cnn_model_epoch_14.pth
#Epoch 15 | Train Loss: 0.1558 | Val Loss: 0.2382 | Val Acc: 92.74%
#Модель сохранена: cnn_model_epoch_15.pth
#Epoch 16 | Train Loss: 0.1523 | Val Loss: 0.2518 | Val Acc: 92.66%
#Модель сохранена: cnn_model_epoch_16.pth
#Epoch 17 | Train Loss: 0.1434 | Val Loss: 0.2441 | Val Acc: 92.74%
#Модель сохранена: cnn_model_epoch_17.pth
#Epoch 18 | Train Loss: 0.1383 | Val Loss: 0.2630 | Val Acc: 92.20%
#Модель сохранена: cnn_model_epoch_18.pth
#Epoch 19 | Train Loss: 0.1343 | Val Loss: 0.2363 | Val Acc: 93.03%
#Модель сохранена: cnn_model_epoch_19.pth
#Epoch 20 | Train Loss: 0.1262 | Val Loss: 0.2461 | Val Acc: 93.03%
#Модель сохранена: cnn_model_epoch_20.pth
#Epoch 21 | Train Loss: 0.1236 | Val Loss: 0.2660 | Val Acc: 92.12%
#Модель сохранена: cnn_model_epoch_21.pth
#Epoch 22 | Train Loss: 0.1218 | Val Loss: 0.2627 | Val Acc: 92.55%
#Модель сохранена: cnn_model_epoch_22.pth
#Epoch 23 | Train Loss: 0.1167 | Val Loss: 0.2479 | Val Acc: 92.63%