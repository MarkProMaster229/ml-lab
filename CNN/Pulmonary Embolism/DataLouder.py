import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import scipy.io
import os
import pydicom 

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