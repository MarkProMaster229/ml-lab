import torch
import matplotlib.pyplot as plt
import DataLouder
import matplotlib.pyplot as plt


import torch.nn as nn
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

MyModel =  AttentionCNNModel()



checkpoint = torch.load("/home/chelovek/BigWork/cnn_model_epoch_9.pth")

MyModel.load_state_dict(checkpoint)
MyModel.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model =  AttentionCNNModel().to(device)
# Убедись, что веса загрузились
print(f"Модель загружена. Режим: {'eval' if not model.training else 'train'}")
trainFix, valid  = DataLouder.get_dataloaders(1)
print("я тут ")
for eposch in range(1):
    MyModel.eval()
    
    for i,(images, masks) in enumerate(valid):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        if masks.sum() > 0:
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