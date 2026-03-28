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

    


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel,self).__init__()
        #self.conv1 = nn.Conv2d(6,64,kernel_size=3,padding=1)
        #self.conv2 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        #self.conv3 = nn.Conv2d(128,256,kernel_size=3,padding=1)
        #self.conv4 = nn.Conv2d(256,512,kernel_size=3,padding=1)
        #self.conv5 = nn.Conv2d(512,1024,kernel_size=3,padding=1)

        #self.final_conv = nn.Conv2d(1024,6,kernel_size=1)
        #self.relu = nn.ReLU()


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

        #Блок 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        #Блок 5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)



#decoder layers
        # ----- ДЕКОДЕР (ИСПРАВЛЕННЫЙ) -----
        # Декодер блок 4: после конкатенации будет 768 каналов
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4_1 = nn.Conv2d(768, 256, kernel_size=3, padding=1)  # 768 = 256(up) + 512(skip) - ИСПРАВЛЕНО!
        self.dec4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Декодер блок 3: после конкатенации будет 384 канала? Давай посчитаем:
        # Из dec4_2 выходит 256 каналов
        # upconv3 превратит 256 -> 128
        # skip3 имеет 256 каналов
        # Значит после конкатенации: 128 + 256 = 384
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3_1 = nn.Conv2d(384, 128, kernel_size=3, padding=1)  # 384 = 128(up) + 256(skip) - ИСПРАВЛЕНО!
        self.dec3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Декодер блок 2: после конкатенации будет 192 канала
        # Из dec3_2 выходит 128 каналов
        # upconv2 превратит 128 -> 64
        # skip2 имеет 128 каналов
        # Значит после конкатенации: 64 + 128 = 192
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2_1 = nn.Conv2d(192, 64, kernel_size=3, padding=1)   # 192 = 64(up) + 128(skip) - ИСПРАВЛЕНО!
        self.dec2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Декодер блок 1: после конкатенации будет 128 каналов
        # Из dec2_2 выходит 64 канала
        # upconv1 превратит 64 -> 64
        # skip1 имеет 64 канала
        # Значит после конкатенации: 64 + 64 = 128
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)   # 128 = 64(up) + 64(skip) - ЭТО БЫЛО ПРАВИЛЬНО!
        self.dec1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Финальная свертка
        self.final_conv = nn.Conv2d(64, 6, kernel_size=1)
        
        # Activation
        self.relu = nn.ReLU(inplace=True)

        



        


    def forward(self, x):
        # Блок 1
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        skip1 = x  # Сохраняем для skip connection
        x = self.pool1(x)
        
        # Блок 2
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        skip2 = x  # Сохраняем для skip connection
        x = self.pool2(x)
        
        # Блок 3
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        skip3 = x  # Сохраняем для skip connection
        x = self.pool3(x)
        
        # Блок 4
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        skip4 = x  # Сохраняем для skip connection
        x = self.pool4(x)
        
        # Блок 5 (bottleneck)
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        # Здесь x имеет размер H/16, W/16, 512 каналов
        
        # ДЕКОДЕР 
        # Декодер блок 4
        x = self.upconv4(x)  # H/8, W/8, 256
        x = torch.cat([x, skip4], dim=1)  # Конкатенация: 256 + 256 = 512
        x = self.relu(self.dec4_1(x))
        x = self.relu(self.dec4_2(x))  # H/8, W/8, 256
        
        # Декодер блок 3
        x = self.upconv3(x)  # H/4, W/4, 128
        x = torch.cat([x, skip3], dim=1)  # 128 + 128 = 256
        x = self.relu(self.dec3_1(x))
        x = self.relu(self.dec3_2(x))  # H/4, W/4, 128
        
        # Декодер блок 2
        x = self.upconv2(x)  # H/2, W/2, 64
        x = torch.cat([x, skip2], dim=1)  # 64 + 64 = 128
        x = self.relu(self.dec2_1(x))
        x = self.relu(self.dec2_2(x))  # H/2, W/2, 64
        
        # Декодер блок 1
        x = self.upconv1(x)  # H, W, 64
        x = torch.cat([x, skip1], dim=1)  # 64 + 64 = 128
        x = self.relu(self.dec1_1(x))
        x = self.relu(self.dec1_2(x))  # H, W, 64
        
        # Финальный выход
        out = self.final_conv(x)  # H, W, 6
        
        return out
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Всего параметров: {total_params:,}")




criterion = CombinedLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
train_loader, val_loader = get_dataloaders(batch_size=3)

ColVo_epoch = 25

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
#2861 step in epoch
        #if i % 25 == 0:
            #print(f"Эпоха [{epoch+1}/{ColVo_epoch}], Шаг {i}, Ошибка: {loss.item():.4f}")


            #print(f"Количество батчей в эпохе: {len(dataloader)}")


        #if epoch >= 2 and i % 100 == 0 and masks.sum() > 0: 
        #    with torch.no_grad():
        #        pred_map = torch.sigmoid(outputs[0, 0]).cpu().numpy()
        #        true_mask = masks[0, 0].cpu().numpy()
        #        
        #        original_img = images[0, 0].cpu().numpy()
        #        
        #    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        #    ax[0].imshow(original_img, cmap='bone')
        #    ax[1].imshow(true_mask, cmap='gray') 
        #    ax[1].set_title(f"Mask (Пикселей: {true_mask.sum()})")
        #    ax[2].imshow(pred_map, cmap='jet')
        #    ax[2].set_title("Prediction Probability")
        #    plt.show()

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


#Всего параметров: 18,630,022
#Эпоха [1/25], Средняя ошибка: 0.9650
#Валидация: Средняя ошибка: 0.9403
#Модель сохранена: cnn_model_epoch_1.pth
#Эпоха [2/25], Средняя ошибка: 0.8257
#Валидация: Средняя ошибка: 0.8836
#Модель сохранена: cnn_model_epoch_2.pth
#Эпоха [3/25], Средняя ошибка: 0.7918
#Валидация: Средняя ошибка: 0.8887
#Модель сохранена: cnn_model_epoch_3.pth
#Эпоха [4/25], Средняя ошибка: 0.7571
#Валидация: Средняя ошибка: 0.8625
#Модель сохранена: cnn_model_epoch_4.pth
#Эпоха [5/25], Средняя ошибка: 0.7316
#Валидация: Средняя ошибка: 0.8713
#Модель сохранена: cnn_model_epoch_5.pth
#Эпоха [6/25], Средняя ошибка: 0.7029
#Валидация: Средняя ошибка: 0.8787
#Модель сохранена: cnn_model_epoch_6.pth
#Эпоха [7/25], Средняя ошибка: 0.7010
#Валидация: Средняя ошибка: 0.8599
#Модель сохранена: cnn_model_epoch_7.pth
#Эпоха [8/25], Средняя ошибка: 0.6846
#Валидация: Средняя ошибка: 0.8981
#Модель сохранена: cnn_model_epoch_8.pth
#Эпоха [9/25], Средняя ошибка: 0.6563
#Валидация: Средняя ошибка: 0.8524
#Модель сохранена: cnn_model_epoch_9.pth
#Эпоха [10/25], Средняя ошибка: 0.6472
#Валидация: Средняя ошибка: 0.8558
#Модель сохранена: cnn_model_epoch_10.pth
#Эпоха [11/25], Средняя ошибка: 0.6320
#Валидация: Средняя ошибка: 0.8551
#Модель сохранена: cnn_model_epoch_11.pth
#Эпоха [12/25], Средняя ошибка: 0.6047
#Валидация: Средняя ошибка: 0.8415
#Модель сохранена: cnn_model_epoch_12.pth
#Эпоха [13/25], Средняя ошибка: 0.6481
#Валидация: Средняя ошибка: 0.3873
#Модель сохранена: cnn_model_epoch_13.pth
#Эпоха [14/25], Средняя ошибка: 0.6764
#Валидация: Средняя ошибка: 0.3864
#Модель сохранена: cnn_model_epoch_14.pth
#Эпоха [15/25], Средняя ошибка: 0.6770
#Валидация: Средняя ошибка: 0.3918
#Модель сохранена: cnn_model_epoch_15.pth
#Эпоха [16/25], Средняя ошибка: 0.6555
#Валидация: Средняя ошибка: 0.3875
#Модель сохранена: cnn_model_epoch_16.pth
#Эпоха [17/25], Средняя ошибка: 0.6807
#Валидация: Средняя ошибка: 0.3867
#Модель сохранена: cnn_model_epoch_17.pth
#Эпоха [18/25], Средняя ошибка: 0.6786
#Валидация: Средняя ошибка: 0.3875
#Модель сохранена: cnn_model_epoch_18.pth
#Эпоха [19/25], Средняя ошибка: 0.6597
#Валидация: Средняя ошибка: 0.3859
#Модель сохранена: cnn_model_epoch_19.pth
#Эпоха [20/25], Средняя ошибка: 0.6628
#Валидация: Средняя ошибка: 0.3865
#Модель сохранена: cnn_model_epoch_20.pth
#Эпоха [21/25], Средняя ошибка: 0.6620
#Валидация: Средняя ошибка: 0.3885
#Модель сохранена: cnn_model_epoch_21.pth
#Эпоха [22/25], Средняя ошибка: 0.6656
#Валидация: Средняя ошибка: 0.3860
#Модель сохранена: cnn_model_epoch_22.pth
#Эпоха [23/25], Средняя ошибка: 0.6684
#Валидация: Средняя ошибка: 0.3863
#Модель сохранена: cnn_model_epoch_23.pth
#Эпоха [24/25], Средняя ошибка: 0.6653
#Валидация: Средняя ошибка: 0.3854
#Модель сохранена: cnn_model_epoch_24.pth
#Эпоха [25/25], Средняя ошибка: 0.6660
#Валидация: Средняя ошибка: 0.3855
#Модель сохранена: cnn_model_epoch_25.pth
