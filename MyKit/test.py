import pydicom
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os

# Настройки
mat_path = '/home/chelovek/BigWork/data/archive/FUMPE/GroundTruth/PAT001.mat'
dcm_dir = '/home/chelovek/BigWork/data/archive/FUMPE/CT_scans/PAT001'
target_slice = 57  # Тот самый срез с тромбом

# Загрузка масок
mat_contents = scipy.io.loadmat(mat_path)
mask_stack = mat_contents['Mask']

# Функция окна (та же, что мы проверили)
def get_windowed_img(dcm_path):
    ds = pydicom.dcmread(dcm_path)
    intercept = ds.RescaleIntercept if 'RescaleIntercept' in ds else 0
    slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1
    image = ds.pixel_array.astype(np.float32) * slope + intercept
    # Окно для сосудов (100, 700)
    img = np.clip(image, 100 - 350, 100 + 350)
    return (img - (100 - 350)) / 700

# Создаем сетку 3x3
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

# Берем срез 57 и по 4 среза в каждую сторону
start_slice = target_slice - 4

for i, ax in enumerate(axes):
    current_idx = start_slice + i
    dcm_filename = f"D{str(current_idx + 1).zfill(4)}.dcm"
    dcm_path = os.path.join(dcm_dir, dcm_filename)
    
    if os.path.exists(dcm_path):
        img = get_windowed_img(dcm_path)
        ax.imshow(img, cmap='bone')
        
        # Если на этом срезе есть разметка, рисуем её контуром
        if np.any(mask_stack[:, :, current_idx] > 0):
            
            ax.set_title(f"Срез {current_idx} (ТРОМБ!)", color='red')
        else:
            ax.set_title(f"Срез {current_idx} (Чисто)")
    
    ax.axis('off')

plt.tight_layout()
plt.show()
