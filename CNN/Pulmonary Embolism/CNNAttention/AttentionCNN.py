from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms
#delete
import os
from Datalouder import get_dataloaders3d
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

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        # W_g: обрабатывает сигнал с нижнего (глубокого) слоя
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # W_x: обрабатывает skip-connection с энкодера
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # psi: вычисляет коэффициенты внимания (маску)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi # Применяем внимание к четким признакам

class AttentionBlock3D(nn.Module):
    """3D версия Attention Gate с автоматическим согласованием размеров"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Сохраняем исходный размер x
        x_shape = x.shape
        
        # Приводим g к размеру x (по глубине, высоте, ширине)
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        # Применяем свертки
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Теперь размеры должны совпадать
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class VGG16_UNet_3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(VGG16_UNet_3D, self).__init__()

        # ---------- Энкодер (VGG-16 стиль) ----------
        # Блок 1: вход (B, 1, 6, H, W)
        self.conv1_1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 6 -> 3

        # Блок 2: вход (B, 64, 3, H/2, W/2)
        self.conv2_1 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 3 -> 1

        # Блок 3: вход (B, 128, 1, H/4, W/4)
        self.conv3_1 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        # Пулим только H,W, глубину не трогаем (kernel_size=1 по глубине)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Блок 4: вход (B, 256, 1, H/8, W/8)
        self.conv4_1 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        # Опять пулим только H,W
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Блок 5 (bottleneck): вход (B, 512, 1, H/16, W/16)
        self.conv5_1 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv3d(512, 512, kernel_size=3, padding=1)

        # ---------- Attention Gates (3D) ----------
        self.att4 = AttentionBlock3D(F_g=256, F_l=512, F_int=256)
        self.att3 = AttentionBlock3D(F_g=128, F_l=256, F_int=128)
        self.att2 = AttentionBlock3D(F_g=64, F_l=128, F_int=64)
        self.att1 = AttentionBlock3D(F_g=64, F_l=64, F_int=32)

        # ---------- Декодер ----------
        # Апскейлы тоже должны соответствовать пулингам
        self.upconv4 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec4_1 = nn.Conv3d(768, 256, kernel_size=3, padding=1)
        self.dec4_2 = nn.Conv3d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec3_1 = nn.Conv3d(384, 128, kernel_size=3, padding=1)
        self.dec3_2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=(3, 2, 2), stride=(3, 2, 2))
        self.dec2_1 = nn.Conv3d(192, 64, kernel_size=3, padding=1)
        self.dec2_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)  # 3 -> 6 по глубине
        self.dec1_1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.dec1_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        skip1 = x
        #print(f"After conv1, x: {x.shape}, skip1: {skip1.shape}")
        x = self.pool1(x)
        #print(f"After pool1: {x.shape}")

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        skip2 = x
        #print(f"After conv2, x: {x.shape}, skip2: {skip2.shape}")
        x = self.pool2(x)
        #print(f"After pool2: {x.shape}")

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        skip3 = x
        #print(f"After conv3, x: {x.shape}, skip3: {skip3.shape}")
        x = self.pool3(x)
        #print(f"After pool3: {x.shape}")

        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        skip4 = x
        #print(f"After conv4, x: {x.shape}, skip4: {skip4.shape}")
        x = self.pool4(x)
        #print(f"After pool4: {x.shape}")

        # Bottleneck
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        #print(f"After bottleneck: {x.shape}")

        # Декодер + Attention
        x = self.upconv4(x)
        #print(f"After upconv4: {x.shape}")
        #print(f"Before att4 - skip4: {skip4.shape}")
        skip4 = self.att4(g=x, x=skip4)
        x = torch.cat([x, skip4], dim=1)
        #print(f"After cat4: {x.shape}")
        x = self.relu(self.dec4_1(x))
        x = self.relu(self.dec4_2(x))
        #print(f"After dec4: {x.shape}")

        x = self.upconv3(x)
        #print(f"After upconv3: {x.shape}")
        #print(f"Before att3 - skip3: {skip3.shape}")
        skip3 = self.att3(g=x, x=skip3)
        x = torch.cat([x, skip3], dim=1)
        #print(f"After cat3: {x.shape}")
        x = self.relu(self.dec3_1(x))
        x = self.relu(self.dec3_2(x))
        #print(f"After dec3: {x.shape}")

        x = self.upconv2(x)
        #print(f"After upconv2: {x.shape}")
        #print(f"Before att2 - skip2: {skip2.shape}")
        skip2 = self.att2(g=x, x=skip2)
        x = torch.cat([x, skip2], dim=1)
        #print(f"After cat2: {x.shape}")
        x = self.relu(self.dec2_1(x))
        x = self.relu(self.dec2_2(x))
        #print(f"After dec2: {x.shape}")

        x = self.upconv1(x)
        #print(f"After upconv1: {x.shape}")
        #print(f"Before att1 - skip1: {skip1.shape}")
        skip1 = self.att1(g=x, x=skip1)
        x = torch.cat([x, skip1], dim=1)
        #print(f"After cat1: {x.shape}")
        x = self.relu(self.dec1_1(x))
        x = self.relu(self.dec1_2(x))
        #print(f"After dec1: {x.shape}")

        out = self.final_conv(x)
        #print(f"Final output: {out.shape}")
        return out.squeeze(1) 
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG16_UNet_3D().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Всего параметров: {total_params:,}")



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

#Эпоха [1/10], Средняя ошибка: 0.9451
#Валидация: Средняя ошибка: 0.8815
#Модель сохранена: cnn_model_epoch_1.pth
#Эпоха [2/10], Средняя ошибка: 0.8940
#Валидация: Средняя ошибка: 0.8626
#Модель сохранена: cnn_model_epoch_2.pth
#Эпоха [3/10], Средняя ошибка: 0.8750
#Валидация: Средняя ошибка: 0.8640
#Модель сохранена: cnn_model_epoch_3.pth
#Эпоха [4/10], Средняя ошибка: 0.8619
#Валидация: Средняя ошибка: 0.8537
#Модель сохранена: cnn_model_epoch_4.pth
#Эпоха [5/10], Средняя ошибка: 0.8470
#Валидация: Средняя ошибка: 0.8409
#Модель сохранена: cnn_model_epoch_5.pth
#Эпоха [6/10], Средняя ошибка: 0.8394
#Валидация: Средняя ошибка: 0.8483
#Модель сохранена: cnn_model_epoch_6.pth
#Эпоха [7/10], Средняя ошибка: 0.8308
#Валидация: Средняя ошибка: 0.8381
##Модель сохранена: cnn_model_epoch_7.pth
#Эпоха [8/10], Средняя ошибка: 0.8312
#Валидация: Средняя ошибка: 0.8577
#Модель сохранена: cnn_model_epoch_8.pth
#Эпоха [9/10], Средняя ошибка: 0.8184
#Валидация: Средняя ошибка: 0.8448
#Модель сохранена: cnn_model_epoch_9.pth
#Эпоха [10/10], Средняя ошибка: 0.8126
#Валидация: Средняя ошибка: 0.8373
#Модель сохранена: cnn_model_epoch_10.pth