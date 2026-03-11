from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms
#delete
import os
from dataLouder3d import get_dataloaders3d
SAVE_DIR = "saved_images"
os.makedirs(SAVE_DIR, exist_ok=True)
#delete
torch.set_num_threads(20)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        # Превращаем логиты в вероятности (0-1)
        predict = torch.sigmoid(predict)
        
        # Сглаживаем в один вектор
        predict = predict.view(-1)
        target = target.view(-1)
        
        intersection = (predict * target).sum()                            
        dice = (2.*intersection + self.smooth) / (predict.sum() + target.sum() + self.smooth)  
        
        return 1 - dice
    

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice(pred, target)

criterion = CombinedLoss()

    
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel3D(nn.Module):
    def __init__(self):
        super(CNNModel3D, self).__init__()

        # ----- ЭНКОДЕР (3D) -----
        
        # Блок 1: вход (B, 1, 6, H, W)
        self.conv1_1 = nn.Conv3d(1, 64, kernel_size=3, padding=1) 
        self.conv1_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        # Уменьшаем глубину: 6 -> 3
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Блок 2: вход (B, 64, 3, H/2, W/2)
        self.conv2_1 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        # Уменьшаем глубину: 3 -> 1 (последнее возможное уменьшение по Z)
        self.pool2 = nn.MaxPool3d(kernel_size=(3, 2, 2), stride=(3, 2, 2))

        # Блок 3: вход (B, 128, 1, H/4, W/4)
        self.conv3_1 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        # Дальше уменьшаем только H и W (глубина 1 остается 1)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Блок 4
        self.conv4_1 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Блок 5 (Bottleneck)
        self.conv5_1 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv3d(512, 512, kernel_size=3, padding=1)

        # ----- ДЕКОДЕР (3D) -----
        
        # Декодер блок 4
        self.upconv4 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec4_1 = nn.Conv3d(768, 256, kernel_size=3, padding=1)
        self.dec4_2 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        
        # Декодер блок 3
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec3_1 = nn.Conv3d(384, 128, kernel_size=3, padding=1)
        self.dec3_2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        
        # Декодер блок 2
        # Восстанавливаем глубину: 1 -> 3
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=(3, 2, 2), stride=(3, 2, 2))
        self.dec2_1 = nn.Conv3d(192, 64, kernel_size=3, padding=1)
        self.dec2_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        
        # Декодер блок 1
        # Восстанавливаем глубину: 3 -> 6
        self.upconv1 = nn.ConvTranspose3d(64, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec1_1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.dec1_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        
        # Финальный выход: возвращаем 1 канал (тромб), глубина 6 сохранится
        self.final_conv = nn.Conv3d(64, 1, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Вход x: (B, 1, 6, H, W)
        
        # Encoder
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        skip1 = x  # (B, 64, 6, H, W)
        x = self.pool1(x)
        
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        skip2 = x  # (B, 128, 3, H/2, W/2)
        x = self.pool2(x)
        
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        skip3 = x  # (B, 256, 1, H/4, W/4)
        x = self.pool3(x)
        
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        skip4 = x  # (B, 512, 1, H/8, W/8)
        x = self.pool4(x)
        
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))

        # Decoder
        x = self.upconv4(x)
        x = torch.cat([x, skip4], dim=1)
        x = self.relu(self.dec4_1(x))
        x = self.relu(self.dec4_2(x))
        
        x = self.upconv3(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.relu(self.dec3_1(x))
        x = self.relu(self.dec3_2(x))
        
        x = self.upconv2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.relu(self.dec2_1(x))
        x = self.relu(self.dec2_2(x))
        
        x = self.upconv1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.relu(self.dec1_1(x))
        x = self.relu(self.dec1_2(x))
        
        out = self.final_conv(x) # (B, 1, 6, H, W)
        return out.squeeze(1) 

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel3D().to(device)



criterion = CombinedLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
train_loader, val_loader = get_dataloaders3d(batch_size=1)

ColVo_epoch = 10
#компромис
accumulation_steps = 3
for epoch in range(ColVo_epoch):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()

    for i,(images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)


        outputs = model(images)
        loss = criterion(outputs, masks)

        loss = loss / accumulation_steps

        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * accumulation_steps  

    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()


    avg_loss = epoch_loss / len(train_loader)
    print(f"Эпоха [{epoch+1}/{ColVo_epoch}], Средняя ошибка: {avg_loss:.4f}")
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images, val_masks = val_images.to(device), val_masks.to(device)
            val_outputs = model(val_images)
            v_loss = criterion(val_outputs, val_masks)
            val_loss += v_loss.item()
            
    print(f"Валидация: Средняя ошибка: {val_loss / len(val_loader):.4f}")

    torch.save(model.state_dict(), f'cnn_model_epoch_{epoch+1}.pth')
    print(f"Модель сохранена: cnn_model_epoch_{epoch+1}.pth")

#Эпоха [1/10], Средняя ошибка: 0.9706
#Валидация: Средняя ошибка: 0.9211
#Модель сохранена: cnn_model_epoch_1.pth
#Эпоха [2/10], Средняя ошибка: 0.8998
#Валидация: Средняя ошибка: 0.8821
#Модель сохранена: cnn_model_epoch_2.pth
#Эпоха [3/10], Средняя ошибка: 0.8862
#Валидация: Средняя ошибка: 0.8667
#Модель сохранена: cnn_model_epoch_3.pth
#Эпоха [4/10], Средняя ошибка: 0.8757
#Валидация: Средняя ошибка: 0.8721
#Модель сохранена: cnn_model_epoch_4.pth
#Эпоха [5/10], Средняя ошибка: 0.8651
#Валидация: Средняя ошибка: 0.8341
#Модель сохранена: cnn_model_epoch_5.pth
#Эпоха [6/10], Средняя ошибка: 0.8562
#: Средняя ошибка: 0.8674
#Модель сохранена: cnn_model_epoch_6.pth
#Эпоха [7/10], Средняя ошибка: 0.8474
#Валидация: Средняя ошибка: 0.8620
#Модель сохранена: cnn_model_epoch_7.pth
# [8/10], Средняя ошибка: 0.8402
#Валидация: Средняя ошибка: 0.8456
#Модель сохранена: cnn_model_epoch_8.pth
#Эпоха [9/10], Средняя ошибка: 0.8336
#Валидация: Средняя ошибка: 0.8183
#Модель сохранена: cnn_model_epoch_9.pth
#Эпоха [10/10], Средняя ошибка: 0.8262
#: Средняя ошибка: 0.8879
#Модель сохранена: cnn_model_epoch_10.pth
