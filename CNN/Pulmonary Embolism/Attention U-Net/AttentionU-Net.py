from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms
#delete
import os
from DataLouder import DataLoader
from DataLouder import get_dataloaders
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

class AttentionCNNModel(nn.Module):
    def __init__(self):
        super(AttentionCNNModel, self).__init__()

        # --- ЭНКОДЕР (Твоя VGG-16 структура) ---
        # Блок 1 
        self.conv1_1 = nn.Conv2d(6, 64, kernel_size=3, padding=1) 
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Блок 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Блок 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Блок 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Блок 5 (Bottleneck)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # --- ATTENTION GATES (Твои каналы) ---
        self.att4 = AttentionBlock(F_g=256, F_l=512, F_int=256)
        self.att3 = AttentionBlock(F_g=128, F_l=256, F_int=128)
        self.att2 = AttentionBlock(F_g=64, F_l=128, F_int=64)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)

        # --- ДЕКОДЕР ---
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4_1 = nn.Conv2d(768, 256, kernel_size=3, padding=1) 
        self.dec4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3_1 = nn.Conv2d(384, 128, kernel_size=3, padding=1) 
        self.dec3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2_1 = nn.Conv2d(192, 64, kernel_size=3, padding=1)  
        self.dec2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  
        self.dec1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.final_conv = nn.Conv2d(64, 6, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Энкодер
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        skip1 = x
        x = self.pool1(x)
        
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        skip2 = x
        x = self.pool2(x)
        
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        skip3 = x
        x = self.pool3(x)
        
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        skip4 = x
        x = self.pool4(x)
        
        # Bottleneck
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        
        # ДЕКОДЕР + ВНИМАНИЕ
        x = self.upconv4(x)
        skip4 = self.att4(g=x, x=skip4) # Накладываем маску
        x = torch.cat([x, skip4], dim=1)
        x = self.relu(self.dec4_1(x))
        x = self.relu(self.dec4_2(x))
        
        x = self.upconv3(x)
        skip3 = self.att3(g=x, x=skip3)
        x = torch.cat([x, skip3], dim=1)
        x = self.relu(self.dec3_1(x))
        x = self.relu(self.dec3_2(x))
        
        x = self.upconv2(x)
        skip2 = self.att2(g=x, x=skip2)
        x = torch.cat([x, skip2], dim=1)
        x = self.relu(self.dec2_1(x))
        x = self.relu(self.dec2_2(x))
        
        x = self.upconv1(x)
        skip1 = self.att1(g=x, x=skip1)
        x = torch.cat([x, skip1], dim=1)
        x = self.relu(self.dec1_1(x))
        x = self.relu(self.dec1_2(x))
        
        out = self.final_conv(x)
        return out

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttentionCNNModel().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Всего параметров: {total_params:,}")



criterion = CombinedLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
train_loader, val_loader = get_dataloaders(batch_size=3)

ColVo_epoch = 10

for epoch in range(ColVo_epoch):
    model.train()
    epoch_loss = 0.0

    for i,(images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)


        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
#2509 step in epoch
        #if i % 25 == 0:
            #print(f"Эпоха [{epoch+1}/{ColVo_epoch}], Шаг {i}, Ошибка: {loss.item():.4f}")

            #print(f"Количество батчей в эпохе: {len(train_loader)}")


        if epoch >= 2 and i % 100 == 0 and masks.sum() > 0: 
            with torch.no_grad():
                pred_map = torch.sigmoid(outputs[0, 0]).cpu().numpy()
                true_mask = masks[0, 0].cpu().numpy()
                
                original_img = images[0, 0].cpu().numpy()
                
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(original_img, cmap='bone')
            ax[1].imshow(true_mask, cmap='gray') 
            ax[1].set_title(f"Mask (Пикселей: {true_mask.sum()})")
            ax[2].imshow(pred_map, cmap='jet')
            ax[2].set_title("Prediction Probability")
            plt.show()

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


# Сравнительный анализ на различных вычислительных узлах подтвердил стабильность архитектуры Attention U-Net. В обоих случаях минимальная ошибка валидации была достигнута в диапазоне 0.846–0.849, что свидетельствует о достижении предела точности на текущем наборе данных (FUMPE)
# 
# гугл коллаб
# Эпоха [1/10], Средняя ошибка: 0.9515
# Валидация: Средняя ошибка: 0.9045
# Модель сохранена: cnn_model_epoch_1.pth
# Эпоха [2/10], Средняя ошибка: 0.8154
# Валидация: Средняя ошибка: 0.8814
# Модель сохранена: cnn_model_epoch_2.pth
# Эпоха [3/10], Средняя ошибка: 0.7900
# Валидация: Средняя ошибка: 0.8959
# Модель сохранена: cnn_model_epoch_3.pth
# Эпоха [4/10], Средняя ошибка: 0.7617
# Валидация: Средняя ошибка: 0.8924
# Модель сохранена: cnn_model_epoch_4.pth
# Эпоха [5/10], Средняя ошибка: 0.7481
# Валидация: Средняя ошибка: 0.8490
# Модель сохранена: cnn_model_epoch_5.pth
# Эпоха [6/10], Средняя ошибка: 0.7245
# Валидация: Средняя ошибка: 0.8693
# Модель сохранена: cnn_model_epoch_6.pth
# Эпоха [7/10], Средняя ошибка: 0.7207
# Валидация: Средняя ошибка: 0.8666
# Модель сохранена: cnn_model_epoch_7.pth
# Эпоха [8/10], Средняя ошибка: 0.6939
# Валидация: Средняя ошибка: 0.8567
# Модель сохранена: cnn_model_epoch_8.pth
# Эпоха [9/10], Средняя ошибка: 0.6652
# Валидация: Средняя ошибка: 0.8559
# Модель сохранена: cnn_model_epoch_9.pth
# Эпоха [10/10], Средняя ошибка: 0.6601
# Валидация: Средняя ошибка: 0.8933
# Модель сохранена: cnn_model_epoch_10.pth
# 
# 
# моя машина 
# 
# 
# Эпоха [1/10], Средняя ошибка: 0.9622
# Валидация: Средняя ошибка: 0.9011
# Модель сохранена: cnn_model_epoch_1.pth
# Эпоха [2/10], Средняя ошибка: 0.8077
# Валидация: Средняя ошибка: 0.8787
# Модель сохранена: cnn_model_epoch_2.pth
# Эпоха [3/10], Средняя ошибка: 0.7757
# Валидация: Средняя ошибка: 0.8779
# Модель сохранена: cnn_model_epoch_3.pth
# Эпоха [4/10], Средняя ошибка: 0.7468
# Валидация: Средняя ошибка: 0.8630
# Модель сохранена: cnn_model_epoch_4.pth
# Эпоха [5/10], Средняя ошибка: 0.7446
# Валидация: Средняя ошибка: 0.8669
# Модель сохранена: cnn_model_epoch_5.pth
# Эпоха [6/10], Средняя ошибка: 0.7160
# Валидация: Средняя ошибка: 0.8544
# Модель сохранена: cnn_model_epoch_6.pth
# Эпоха [7/10], Средняя ошибка: 0.6911
# Валидация: Средняя ошибка: 0.8574
# Модель сохранена: cnn_model_epoch_7.pth
# Эпоха [8/10], Средняя ошибка: 0.6712
# Валидация: Средняя ошибка: 0.8516
# Модель сохранена: cnn_model_epoch_8.pth
# Эпоха [9/10], Средняя ошибка: 0.6582
# Валидация: Средняя ошибка: 0.8464
# Модель сохранена: cnn_model_epoch_9.pth
# Эпоха [10/10], Средняя ошибка: 0.6524
# Валидация: Средняя ошибка: 0.8642
# Модель сохранена: cnn_model_epoch_10.pth
# 
# 
# 
# Vanilla U-Net: Нестабильна, высокий лосс, сильный шум.
# Attention U-Net (Run 1): Лучший результат (0.8464), плавная сходимость.
# Attention U-Net (Run 2): Подтверждение стабильности, быстрая сходимость (0.8490 на 5-й эпохе).
# 
# 
# 
# 
#     «Модификация архитектуры U-Net блоками внимания (Attention Gates) позволила достичь минимального значения функции потерь 0.8464, что превосходит результаты базовой модели».
#     «Внедрение механизмов селективной фильтрации признаков обеспечило более стабильную сходимость: в то время как базовая модель демонстрировала стохастические колебания, Attention U-Net сохраняла тренд на снижение ошибки валидации вплоть до 9-й итерации».
#     «Наблюдаемый всплеск ошибки на 10-й эпохе (с 0.846 до 0.864) указывает на предел емкости модели при текущем объеме выборки (35 пациентов), что обосновывает необходимость применения методов регуляризации на финальных стадиях обучения».