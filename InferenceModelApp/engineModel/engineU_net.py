from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import gc
from flask import Flask, request, jsonify, render_template_string, render_template
from flask_cors import CORS

#this file include engine only

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


#CNN engine
#CNN letter
#------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
from huggingface_hub import hf_hub_download
import numpy as np
import os
class EngineU_net:
    """
    Единый класс для инференса U-Net подобной модели сегментации.
    6 входных каналов (срезы КТ), 6 выходных (маски).
    """
    
    # ========================================================================
    # ВЛОЖЕННЫЕ КЛАССЫ АРХИТЕКТУРЫ
    # ========================================================================
    
    class DiceLoss(nn.Module):
        def __init__(self, smooth=1e-6):
            super().__init__()
            self.smooth = smooth

        def forward(self, predict, target):
            predict = torch.sigmoid(predict)
            predict = predict.view(-1)
            target = target.view(-1)
            intersection = (predict * target).sum()
            dice = (2.*intersection + self.smooth) / (predict.sum() + target.sum() + self.smooth)
            return 1 - dice

    class CombinedLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.bce = nn.BCEWithLogitsLoss()
            self.dice = EngineU_net.DiceLoss()

        def forward(self, pred, target):
            return self.bce(pred, target) + self.dice(pred, target)

    class UNetSegmentation(nn.Module):
        """U-Net подобная модель для сегментации с 6 входными и 6 выходными каналами"""
        
        def __init__(self, in_channels=6, out_channels=6):
            super().__init__()
            
            # ЭНКОДЕР
            # Блок 1
            self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
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

            # Блок 5 (bottleneck)
            self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

            # ДЕКОДЕР
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
            
            self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
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
            
            # Декодер
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
            
            out = self.final_conv(x)
            return out

    # ========================================================================
    # ОСНОВНОЙ КЛАСС ENGINE
    # ========================================================================
    
    def __init__(
        self, 
        repo_id: str = "MarkProMaster229/experimental_models",
        config_path: str = "vanila2dEmbol/config.json",
        weights_path: str = "vanila2dEmbol/cnn_model_epoch_4.pth",
        n_slices: int = 6
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_slices = n_slices
        
        # 1. Скачиваем config.json
        print(f"📥 Загрузка конфига: {repo_id}/{config_path}")
        config_file = hf_hub_download(repo_id=repo_id, filename=config_path)
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 2. Создаём модель
        self.model = self.UNetSegmentation(
            in_channels=self.config.get("in_channels", 6),
            out_channels=self.config.get("out_channels", 6)
        ).to(self.device)
        
        # 3. Грузим веса
        print(f"📥 Загрузка весов: {repo_id}/{weights_path}")
        weights_file = hf_hub_download(repo_id=repo_id, filename=weights_path)
        state_dict = torch.load(weights_file, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        
        self.model.eval()
        print(f"✅ Engine (U-Net Segmentation) загружен")
        print(f"   Модель: {self.config.get('model_type', 'UNetSegmentation')}")
        print(f"   Входных каналов: {self.config.get('in_channels', 6)}")
        print(f"   Выходных каналов: {self.config.get('out_channels', 6)}")
        print(f"   Устройство: {self.device}")
    
    def _window_ct(self, pixel_array, slope=1, intercept=0):
        """Применение КТ-окна к одному срезу"""
        image = pixel_array.astype(np.float32) * slope + intercept
        img = np.clip(image, -250, 450)
        return (img - (-250)) / 700
    
    def predict_from_dicom_series(self, dcm_dir: str):
        """
        Предсказание для серии DICOM-файлов из папки.
        
        Args:
            dcm_dir: путь к папке с DICOM-файлами (.dcm)
            
        Returns:
            numpy.ndarray: маска сегментации формы (H, W, 6)
        """
        import pydicom
        
        dcm_files = sorted([f for f in os.listdir(dcm_dir) if f.endswith('.dcm')])
        
        if len(dcm_files) < self.n_slices:
            raise ValueError(f"Нужно минимум {self.n_slices} срезов, найдено {len(dcm_files)}")
        
        # Берём центральные срезы
        start_idx = (len(dcm_files) - self.n_slices) // 2
        
        slices = []
        for i in range(start_idx, start_idx + self.n_slices):
            dcm_path = os.path.join(dcm_dir, dcm_files[i])
            ds = pydicom.dcmread(dcm_path)
            
            intercept = ds.RescaleIntercept if 'RescaleIntercept' in ds else 0
            slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1
            
            windowed = self._window_ct(ds.pixel_array, slope, intercept)
            slices.append(windowed)
        
        # Стек: (6, H, W)
        input_tensor = torch.tensor(np.stack(slices, axis=0), dtype=torch.float32)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)  # (1, 6, H, W)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            output = torch.sigmoid(output)  # в вероятности 0-1
        
        # (1, 6, H, W) -> (H, W, 6)
        mask = output.squeeze(0).cpu().numpy()
        mask = np.transpose(mask, (1, 2, 0))
        
        return mask
    
    def predict_from_numpy(self, ct_volume: np.ndarray):
        """
        Предсказание для numpy-массива срезов КТ.
        
        Args:
            ct_volume: numpy-массив формы (n_slices, H, W)
            
        Returns:
            numpy.ndarray: маска сегментации формы (H, W, 6)
        """
        if ct_volume.shape[0] != self.n_slices:
            raise ValueError(f"Ожидалось {self.n_slices} срезов, получено {ct_volume.shape[0]}")
        
        input_tensor = torch.tensor(ct_volume, dtype=torch.float32)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            output = torch.sigmoid(output)
        
        mask = output.squeeze(0).cpu().numpy()
        mask = np.transpose(mask, (1, 2, 0))
        
        return mask
    
    def predict_binary_masks(self, ct_volume: np.ndarray, threshold: float = 0.5):
        """
        Возвращает бинарные маски по порогу.
        
        Returns:
            dict: словарь с масками для каждого класса
        """
        prob_mask = self.predict_from_numpy(ct_volume)
        binary_mask = (prob_mask > threshold).astype(np.uint8)
        
        return {
            "probabilities": prob_mask,
            "binary": binary_mask,
            "threshold": threshold
        }
        
    def visualize_sequence(self, dcm_dir: str, mat_path: str = None, threshold: float = 0.5):
        import matplotlib.pyplot as plt
        import pydicom
        import scipy.io
        
        # Загружаем Ground Truth
        if mat_path and os.path.exists(mat_path):
            gt_mask_original = scipy.io.loadmat(mat_path)['Mask']  # (H, W, N)
            has_gt = True
        else:
            gt_mask_original = None
            has_gt = False
        
        # Загружаем DICOM
        dcm_files = sorted([f for f in os.listdir(dcm_dir) if f.endswith('.dcm')])
        start_idx = (len(dcm_files) - self.n_slices) // 2
        
        ct_slices_original = []
        for i in range(start_idx, start_idx + self.n_slices):
            dcm_path = os.path.join(dcm_dir, dcm_files[i])
            ds = pydicom.dcmread(dcm_path)
            intercept = ds.RescaleIntercept if 'RescaleIntercept' in ds else 0
            slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1
            windowed = self._window_ct(ds.pixel_array, slope, intercept)
            ct_slices_original.append(windowed)
        
        # Текущее состояние
        ct_slices = [s.copy() for s in ct_slices_original]
        gt_mask = gt_mask_original.copy() if has_gt else None
        
        def apply_transform(transform_type):
            nonlocal ct_slices, gt_mask, pred_mask, pred_binary
            
            if transform_type == 'rotate':
                ct_slices = [np.rot90(s, 2) for s in ct_slices]
                if has_gt:
                    gt_mask = np.rot90(gt_mask, 2, axes=(0,1))  # Трансформируем ВСЮ 3D маску
                print("🔄 Применён ПОВОРОТ 180°")
            elif transform_type == 'shift':
                ct_slices = [np.roll(s, 50, axis=1) for s in ct_slices]
                if has_gt:
                    gt_mask = np.roll(gt_mask, 50, axis=1)  # Трансформируем ВСЮ 3D маску
                print("➡️ Применён СДВИГ ВПРАВО на 50px")
            elif transform_type == 'flip':
                ct_slices = [np.fliplr(s) for s in ct_slices]
                if has_gt:
                    gt_mask = np.fliplr(gt_mask)  # Трансформируем ВСЮ 3D маску
                print("🪞 Применёно ЗЕРКАЛО")
            elif transform_type == 'reset':
                ct_slices = [s.copy() for s in ct_slices_original]
                gt_mask = gt_mask_original.copy() if has_gt else None
                print("🔄 Сброс к оригиналу")
            
            # Пересчитываем предсказания
            ct_volume = np.stack(ct_slices, axis=0)
            pred_mask = self.predict_from_numpy(ct_volume)
            pred_binary = (pred_mask > threshold).astype(np.uint8)
        
        # Начальное предсказание
        ct_volume = np.stack(ct_slices, axis=0)
        pred_mask = self.predict_from_numpy(ct_volume)
        pred_binary = (pred_mask > threshold).astype(np.uint8)
        
        # Визуализация
        fig, axes = plt.subplots(1, 3 if has_gt else 2, figsize=(15, 5))
        if not has_gt:
            axes = [axes[0], axes[1]]
        
        colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1)]
        
        current_idx = 0
        current_transform = "ОРИГИНАЛ"
        
        def update_display(idx):
            for ax in axes:
                ax.clear()
            
            # КТ
            axes[0].imshow(ct_slices[idx], cmap='gray')
            axes[0].set_title(f'КТ срез {idx+1}/6 [{current_transform}]')
            axes[0].axis('off')
            
            # Ground Truth
            if has_gt and gt_mask is not None:
                gt_slice_idx = start_idx + idx
                if gt_slice_idx < gt_mask.shape[2]:
                    gt_slice = gt_mask[:, :, gt_slice_idx]
                    
                    gt_overlay = np.zeros((*ct_slices[idx].shape, 3))
                    ct_norm = (ct_slices[idx] - ct_slices[idx].min()) / (ct_slices[idx].max() - ct_slices[idx].min() + 1e-8)
                    gt_overlay[:, :, 0] = ct_norm
                    gt_overlay[:, :, 1] = ct_norm
                    gt_overlay[:, :, 2] = ct_norm
                    
                    for c in range(1, 6):
                        if c <= gt_slice.max():
                            class_mask = (gt_slice == c)
                            if class_mask.sum() > 0:
                                for ch in range(3):
                                    gt_overlay[:, :, ch] = np.where(
                                        class_mask,
                                        gt_overlay[:, :, ch] * 0.5 + colors[c-1][ch] * 0.5,
                                        gt_overlay[:, :, ch]
                                    )
                    axes[1].imshow(gt_overlay)
                    axes[1].set_title(f'GT (трансформирована)')
                axes[1].axis('off')
            
            # Предсказание
            pred_ax = axes[2] if has_gt else axes[1]
            pred_overlay = np.zeros((*ct_slices[idx].shape, 3))
            ct_norm = (ct_slices[idx] - ct_slices[idx].min()) / (ct_slices[idx].max() - ct_slices[idx].min() + 1e-8)
            pred_overlay[:, :, 0] = ct_norm
            pred_overlay[:, :, 1] = ct_norm
            pred_overlay[:, :, 2] = ct_norm
            
            for c in range(1, 6):
                if pred_binary[:, :, c].sum() > 0:
                    for ch in range(3):
                        pred_overlay[:, :, ch] = np.where(
                            pred_binary[:, :, c] > 0,
                            pred_overlay[:, :, ch] * 0.5 + colors[c-1][ch] * 0.5,
                            pred_overlay[:, :, ch]
                        )
            pred_ax.imshow(pred_overlay)
            pred_ax.set_title(f'Pred (th={threshold})')
            pred_ax.axis('off')
            
            fig.suptitle(f'← → навигация | T:поворот | S:сдвиг | F:зеркало | R:сброс', fontsize=12)
            fig.canvas.draw()
        
        def on_key(event):
            nonlocal current_idx, current_transform
            if event.key == 'left':
                current_idx = max(0, current_idx - 1)
                update_display(current_idx)
            elif event.key == 'right':
                current_idx = min(5, current_idx + 1)
                update_display(current_idx)
            elif event.key == 't':
                current_transform = "ПОВОРОТ 180°"
                apply_transform('rotate')
                update_display(current_idx)
            elif event.key == 's':
                current_transform = "СДВИГ ВПРАВО"
                apply_transform('shift')
                update_display(current_idx)
            elif event.key == 'f':
                current_transform = "ЗЕРКАЛО"
                apply_transform('flip')
                update_display(current_idx)
            elif event.key == 'r':
                current_transform = "ОРИГИНАЛ"
                apply_transform('reset')
                update_display(current_idx)
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        update_display(0)
        
        print("\n" + "="*60)
        print("🎮 УПРАВЛЕНИЕ:")
        print("   ← →  : переключение срезов")
        print("   T    : ПОВОРОТ 180°")
        print("   S    : СДВИГ ВПРАВО")
        print("   F    : ЗЕРКАЛО")
        print("   R    : СБРОС")
        print("="*60 + "\n")
        
        plt.show()
    @staticmethod
    def demo():
        """
        Демо: проходит по пациентам 33, 34, 35 с интерактивной визуализацией
        """
        import kagglehub
        import os
        import glob
        
        print("📥 Загрузка датасета...")
        dataset_path = kagglehub.dataset_download("andrewmvd/pulmonary-embolism-in-ct-images")
        
        # Ищем все папки с DICOM
        all_patients = {}
        for root, dirs, files in os.walk(dataset_path):
            dcm_files = [f for f in files if f.endswith('.dcm')]
            if len(dcm_files) > 100:
                patient_id = os.path.basename(root)
                all_patients[patient_id] = root
        
        # Сортируем и выбираем 33, 34, 35 (если есть)
        target_patients = ['PAT033', 'PAT034', 'PAT035']
        selected = {}
        for pid in target_patients:
            if pid in all_patients:
                selected[pid] = all_patients[pid]
            else:
                # Ищем любые другие, если нет точных совпадений
                matching = [p for p in all_patients.keys() if pid in p or pid.replace('0', '') in p]
                if matching:
                    selected[pid] = all_patients[matching[0]]
        
        if not selected:
            # Берём первых трёх попавшихся
            for i, (pid, path) in enumerate(list(all_patients.items())[:3]):
                selected[f"пациент_{i+1}"] = path
        
        print(f"🤖 Загрузка модели...")
        engine = EngineU_net()
        
        for patient_name, dcm_dir in selected.items():
            patient_id = os.path.basename(dcm_dir)
            
            # Ищем .mat файл
            mat_path = None
            search_paths = [
                os.path.join(dataset_path, f"{patient_id}.mat"),
                os.path.join(os.path.dirname(dcm_dir), f"{patient_id}.mat"),
                os.path.join(dataset_path, "GroundTruth", f"{patient_id}.mat"),
            ]
            
            for path in search_paths:
                if os.path.exists(path):
                    mat_path = path
                    break
            
            if mat_path is None:
                mat_files = glob.glob(os.path.join(dataset_path, "**", "*.mat"), recursive=True)
                for mf in mat_files:
                    if patient_id in mf:
                        mat_path = mf
                        break
            
            print("\n" + "="*60)
            print(f"📂 ПАЦИЕНТ: {patient_id}")
            print("="*60)
            print(f"📁 DICOM: {dcm_dir}")
            if mat_path:
                print(f"📋 Разметка: {mat_path}")
            else:
                print(f"⚠️ Разметка не найдена")
            
            print(f"\n🎮 УПРАВЛЕНИЕ:")
            print("   ← → срезы | T:поворот 180° | S:сдвиг | F:зеркало | R:сброс")
            print("   Закройте окно для перехода к следующему пациенту\n")
            
            engine.visualize_sequence(
                dcm_dir=dcm_dir, 
                mat_path=mat_path, 
                threshold=0.5
            )
            
            # Спрашиваем, продолжать ли
            if patient_name != list(selected.keys())[-1]:
                response = input(f"\n✅ Пациент {patient_id} просмотрен. Продолжить? (Y/n): ")
                if response.lower() == 'n':
                    print("👋 Демо завершено.")
                    return
        
        print("\n🎉 Все пациенты просмотрены!")


    def evaluate_invariance2(self, dcm_dir: str, mat_path: str = None, threshold: float = 0.5):
        """
        Количественная оценка пространственной инвариантности для 2D моделей.
        Выход: (H, W, 6) — 6 каналов (классы)
        """
        import pydicom
        import scipy.io
        
        print("\n" + "="*70)
        print("📊 КОЛИЧЕСТВЕННАЯ ОЦЕНКА ПРОСТРАНСТВЕННОЙ ИНВАРИАНТНОСТИ (2D)")
        print("="*70)
        
        # Загружаем данные
        dcm_files = sorted([f for f in os.listdir(dcm_dir) if f.endswith('.dcm')])
        start_idx = (len(dcm_files) - self.n_slices) // 2
        
        ct_slices = []
        for i in range(start_idx, start_idx + self.n_slices):
            dcm_path = os.path.join(dcm_dir, dcm_files[i])
            ds = pydicom.dcmread(dcm_path)
            intercept = ds.RescaleIntercept if 'RescaleIntercept' in ds else 0
            slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1
            windowed = self._window_ct(ds.pixel_array, slope, intercept)
            ct_slices.append(windowed)
        
        ct_original = np.stack(ct_slices, axis=0)  # (6, H, W)
        
        # Базовое предсказание — форма (H, W, 6)
        pred_original = self.predict_from_numpy(ct_original)
        # Суммируем по классам 1-5 (класс 0 — фон)
        binary_original = (pred_original[:, :, 1:].sum(axis=2) > threshold).astype(np.uint8)
        total_pixels_original = binary_original.sum()
        
        print(f"\n📌 ОРИГИНАЛ: {total_pixels_original} активированных пикселей (100%)")
        
        results = {}
        
        # ========== ТЕСТ 1: ПОВОРОТ 180° ==========
        ct_rotated = np.stack([np.rot90(s, 2) for s in ct_slices], axis=0)
        pred_rotated = self.predict_from_numpy(ct_rotated)  # (H, W, 6)
        
        # Поворачиваем предсказание обратно
        pred_rotated_back = np.rot90(pred_rotated, 2, axes=(0, 1))  # (H, W, 6)
        binary_rotated_back = (pred_rotated_back[:, :, 1:].sum(axis=2) > threshold).astype(np.uint8)
        
        pixels_rotated = (pred_rotated[:, :, 1:].sum(axis=2) > threshold).sum()
        pixels_rotated_back = binary_rotated_back.sum()
        
        intersection = (binary_original & binary_rotated_back).sum()
        union = (binary_original | binary_rotated_back).sum()
        iou = intersection / union if union > 0 else 0
        
        results['rotate'] = {
            'name': 'Поворот 180°',
            'pixels_after': pixels_rotated,
            'pixels_after_back': pixels_rotated_back,
            'retention': (pixels_rotated_back / total_pixels_original * 100) if total_pixels_original > 0 else 0,
            'iou': iou * 100
        }
        
        # ========== ТЕСТ 2: СДВИГ ==========
        shift_values = [10, 25, 50, 75, 100]
        results['shifts'] = []
        
        for shift_px in shift_values:
            ct_shifted = np.stack([np.roll(s, shift_px, axis=1) for s in ct_slices], axis=0)
            pred_shifted = self.predict_from_numpy(ct_shifted)  # (H, W, 6)
            
            # Сдвигаем обратно
            pred_shifted_back = np.roll(pred_shifted, -shift_px, axis=1)
            binary_shifted_back = (pred_shifted_back[:, :, 1:].sum(axis=2) > threshold).astype(np.uint8)
            
            pixels_shifted = (pred_shifted[:, :, 1:].sum(axis=2) > threshold).sum()
            pixels_shifted_back = binary_shifted_back.sum()
            
            intersection = (binary_original & binary_shifted_back).sum()
            union = (binary_original | binary_shifted_back).sum()
            iou = intersection / union if union > 0 else 0
            
            results['shifts'].append({
                'pixels': shift_px,
                'pixels_after': pixels_shifted,
                'pixels_after_back': pixels_shifted_back,
                'retention': (pixels_shifted_back / total_pixels_original * 100) if total_pixels_original > 0 else 0,
                'iou': iou * 100
            })
        
        # ========== ТЕСТ 3: ЗЕРКАЛО ==========
        ct_flipped = np.stack([np.fliplr(s) for s in ct_slices], axis=0)
        pred_flipped = self.predict_from_numpy(ct_flipped)  # (H, W, 6)
        
        pred_flipped_back = np.fliplr(pred_flipped)
        binary_flipped_back = (pred_flipped_back[:, :, 1:].sum(axis=2) > threshold).astype(np.uint8)
        
        pixels_flipped = (pred_flipped[:, :, 1:].sum(axis=2) > threshold).sum()
        pixels_flipped_back = binary_flipped_back.sum()
        
        intersection = (binary_original & binary_flipped_back).sum()
        union = (binary_original | binary_flipped_back).sum()
        iou = intersection / union if union > 0 else 0
        
        results['flip'] = {
            'name': 'Зеркало',
            'pixels_after': pixels_flipped,
            'pixels_after_back': pixels_flipped_back,
            'retention': (pixels_flipped_back / total_pixels_original * 100) if total_pixels_original > 0 else 0,
            'iou': iou * 100
        }
        
        # ========== ВЫВОД РЕЗУЛЬТАТОВ ==========
        print("\n" + "-"*70)
        print("📈 РЕЗУЛЬТАТЫ ТЕСТОВ:")
        print("-"*70)
        
        r = results['rotate']
        print(f"\n🔄 ПОВОРОТ 180°:")
        print(f"   Активаций: {r['pixels_after']} px")
        print(f"   После обратного поворота: {r['pixels_after_back']} px")
        print(f"   Сохранение: {r['retention']:.1f}%")
        print(f"   IoU с оригиналом: {r['iou']:.1f}%")
        
        print(f"\n➡️ СДВИГ ВПРАВО:")
        print(f"   {'Сдвиг':<8} {'Активаций':<12} {'После возврата':<14} {'Сохранение':<12} {'IoU':<8}")
        print(f"   {'-'*60}")
        for s in results['shifts']:
            print(f"   {s['pixels']:3d} px   {s['pixels_after']:5d} px     {s['pixels_after_back']:5d} px        {s['retention']:5.1f}%      {s['iou']:5.1f}%")
        
        r = results['flip']
        print(f"\n🪞 ЗЕРКАЛО:")
        print(f"   Активаций: {r['pixels_after']} px")
        print(f"   После обратного отражения: {r['pixels_after_back']} px")
        print(f"   Сохранение: {r['retention']:.1f}%")
        print(f"   IoU с оригиналом: {r['iou']:.1f}%")
        
        # ========== ИТОГОВАЯ ТАБЛИЦА ==========
        print("\n" + "="*70)
        print("📋 СВОДНАЯ ТАБЛИЦА ДЛЯ ПУБЛИКАЦИИ:")
        print("="*70)
        print(f"""
            Модель: 2D U_net
            Оригинальных активаций: {total_pixels_original} px
            
            ┌─────────────────┬──────────────┬──────────────┬─────────────┐
            │ Трансформация   │ Сохранение   │ IoU          │ Статус      │
            ├─────────────────┼──────────────┼──────────────┼─────────────┤
            │ Поворот 180°    │ {results['rotate']['retention']:5.1f}%       │ {results['rotate']['iou']:5.1f}%
            │ Сдвиг 10px      │ {results['shifts'][0]['retention']:5.1f}%       │ {results['shifts'][0]['iou']:5.1f}%
            │ Сдвиг 25px      │ {results['shifts'][1]['retention']:5.1f}%       │ {results['shifts'][1]['iou']:5.1f}%
            │ Сдвиг 50px      │ {results['shifts'][2]['retention']:5.1f}%       │ {results['shifts'][2]['iou']:5.1f}%
            │ Сдвиг 75px      │ {results['shifts'][3]['retention']:5.1f}%       │ {results['shifts'][3]['iou']:5.1f}% 
            │ Сдвиг 100px     │ {results['shifts'][4]['retention']:5.1f}%       │ {results['shifts'][4]['iou']:5.1f}%
            │ Зеркало         │ {results['flip']['retention']:5.1f}%       │ {results['flip']['iou']:5.1f}%
            └─────────────────┴──────────────┴──────────────┴─────────────┘
            """)
        
        return results


y = EngineU_net()

y.demo()
y.evaluate_invariance2(
        dcm_dir="/home/chelovek/Рабочий стол/PATFully/PAT034",
        mat_path="/home/chelovek/Рабочий стол/PATFully/PAT034.mat",
        threshold=0.5
)
