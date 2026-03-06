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
from DataLouder import create_full_dataset
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
        self.conv1 = nn.Conv2d(6,64,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(256,512,kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(512,1024,kernel_size=3,padding=1)

        self.final_conv = nn.Conv2d(1024,6,kernel_size=1)
        self.relu = nn.ReLU()


        # Блок 1 
        self.conv1_1 = nn.Conv2d(6, 64, kernel_size=3, padding=1) 
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Блок 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        #self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Блок 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        #self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        #Блок 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        #self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        #Блок 5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        #self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)





        


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))


        out = self.final_conv(x) 
        return out
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel().to(device)



criterion = CombinedLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
datalouderChankCut = create_full_dataset()
dataloader = DataLoader(datalouderChankCut, batch_size=1, shuffle=True)

ColVo_epoch = 10

for epoch in range(ColVo_epoch):
    model.train()

    for i,(images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)


        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad() # обнуляем старые градиенты
        loss.backward()      # считаем новые
        optimizer.step()     # делаем шаг обучения

        if i % 5 == 0:
            print(f"Эпоха [{epoch+1}/{ColVo_epoch}], Шаг {i}, Ошибка: {loss.item():.4f}")

        if i % 20 == 0 and masks.sum() > 0: 
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


