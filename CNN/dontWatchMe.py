from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms
#delete
import os


SAVE_DIR = "saved_images"
os.makedirs(SAVE_DIR, exist_ok=True)
#delete
torch.set_num_threads(20)


import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import scipy.io
import os
import pydicom 


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

class DataLouder(Dataset):
    def __init__(self, dcm_dir, mat_path, n_slices=6):
        self.dcm_dir = dcm_dir
        self.mat_path = mat_path
        self.n_slices = n_slices
        # Загружаем маску один раз при инициализации пациента
        self.mask_stack = scipy.io.loadmat(mat_path)['Mask']
        self.dcm_files = sorted([f for f in os.listdir(dcm_dir) if f.endswith('.dcm')])

    def __len__(self):
        return len(self.dcm_files) - self.n_slices
    
    def get_windowed_img(self, dcm_path):
        ds = pydicom.dcmread(dcm_path)
        intercept = ds.RescaleIntercept if 'RescaleIntercept' in ds else 0
        slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1
        image = ds.pixel_array.astype(np.float32) * slope + intercept
        # Окно (100, 700) -> [-250, 450]
        img = np.clip(image, -250, 450)
        return (img - (-250)) / 700

    def __getitem__(self, idx):
        slices = []
        for i in range(idx, idx + self.n_slices):
            img_path = os.path.join(self.dcm_dir, self.dcm_files[i])
            slices.append(self.get_windowed_img(img_path))
        
        full_input = torch.tensor(np.stack(slices, axis=0), dtype=torch.float32)
        
        # Берем 6 масок и разворачиваем каналы вперед
        target_masks = self.mask_stack[:, :, idx : idx + self.n_slices]
        target_tensor = torch.tensor(np.transpose(target_masks, (2, 0, 1)), dtype=torch.float32)
        
        return full_input, target_tensor

def create_full_dataset(base_ct_dir, base_mask_dir):
    all_datasets = []
    patient_ids = sorted([d for d in os.listdir(base_ct_dir) if d.startswith('PAT')])
    for pat_id in patient_ids:
        dcm_path = os.path.join(base_ct_dir, pat_id)
        mat_file = os.path.join(base_mask_dir, f"{pat_id}.mat")
        if os.path.exists(mat_file):
            all_datasets.append(DataLouder(dcm_dir=dcm_path, mat_path=mat_file))
    return ConcatDataset(all_datasets)


def get_dataloaders(batch_size=1):
    train_ds = create_full_dataset(
        '/home/chelovek/bigWork/FUMPE/CT_scans',
        '/home/chelovek/bigWork/FUMPE/GroundTruth'
    )
    val_ds = create_full_dataset(
        '/home/chelovek/bigWork/Valid/Image/',
        '/home/chelovek/bigWork/Valid/map/'
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader



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


#мне просто стало интересно, может я всех действительно всех обманываю. Даже сам себя.

class IDolbaeb(nn.Module):
    def __init__(self):
        super(IDolbaeb,self).__init__()
        self.conv1_1 = nn.Conv2d(6, 64, kernel_size=3, padding=1) 
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(128, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)



        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)

                #over!
        self.MergeAttention = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.dec4_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dec4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv5 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=256, F_l=256, F_int=256)

        self.dec2_2 = nn.Conv2d(256, 64, kernel_size=3, padding=1)

        self.final_conv = nn.Conv2d(64, 6, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        #print("SHAPE:", x.shape)
        x = self.relu(self.conv1_2(x))
        #print("SHAPE:", x.shape)
        x = self.pool1(x)
        #print("SHAPE:", x.shape)

        x = self.relu(self.conv2_1(x))
        #print("SHAPE:", x.shape)
        x = self.relu(self.conv2_2(x))
        #print("SHAPE:", x.shape)
        x = self.pool2(x)
        #print("SHAPE:", x.shape)

        x = self.relu(self.conv5_1(x))
        #print("SHAPE:", x.shape)
        x = self.relu(self.conv5_2(x))
        #print("SHAPE:", x.shape)

        x = self.upconv4(x)
        #print("SHAPE:", x.shape)
        skipConnToAtten = x
        #идея - законкатится с тензором до внимания и после внимания глянем что с вниманием 
        #TODO да внимание живо я получил планомерный спад на несколько эпох при замороженых весах основной сети 
        x = self.att4(g = x, x = x)

        x = self.att4(x,x)
        #print("SHAPE:", x.shape)
        x = torch.cat([x,skipConnToAtten], dim=1)
        x = self.MergeAttention(x)
        #print("SHAPE:", x.shape)
        x = self.relu(self.dec4_1(x))
        

        #print("SHAPE:", x.shape)
        x = self.relu(self.dec4_2(x))

        #print("SHAPE:", x.shape)
        x = self.relu(self.dec4_2(x))
        #print("SHAPE:", x.shape)


        x = self.upconv5(x)
        #print("SHAPE:", x.shape)

        x = self.relu(self.dec2_2(x))
        #print("SHAPE:", x.shape)
        x = self.final_conv(x)
        #print("SHAPE:", x.shape)

        return x 
weighLoad = torch.load("/home/chelovek/bigWork/cnn_model_epoch_4ThisShitModel.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = IDolbaeb().to(device)
model.load_state_dict(torch.load("/home/chelovek/bigWork/cnn_model_epoch_4ThisShitModel.pth"),strict=False)

model.conv1_1.requires_grad_(False)
model.conv1_2.requires_grad_(False)
model.pool1.requires_grad_(False)
model.conv2_1.requires_grad_(False)
model.conv2_2.requires_grad_(False)
model.pool2.requires_grad_(False)
model.conv5_1.requires_grad_(False)
model.conv5_2.requires_grad_(False)
model.upconv4.requires_grad_(False)
model.dec4_1.requires_grad_(False)
model.dec4_2.requires_grad_(False)
model.upconv5.requires_grad_(False)
model.dec2_2.requires_grad_(False)
model.final_conv.requires_grad_(False)

#model.dec4_1.requires_grad_(False)

total_params = sum(p.numel() for p in model.parameters())
print(f"Всего параметров: {total_params:,}")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"параметров: {trainable_params:,}")


frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print("frozen:", frozen)
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

    torch.save(model.state_dict(), f'cnn_model_epoch_{epoch+1}ThisShitModelAttention.pth')