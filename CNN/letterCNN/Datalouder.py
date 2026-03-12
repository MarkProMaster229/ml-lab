from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
import re
from PIL import Image, ImageOps
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
import re
from PIL import Image, ImageOps
import numpy as np

class FontLetterDataset(Dataset):
    def __init__(self, split="train", img_size=224, test_size=None, seed=42):
        self.img_size = img_size
        self.test_size = test_size
        self.seed = seed

        # Загружаем датасет (он будет в memory‑mapped формате)
        self._load_dataset(split)
        self._create_alphabet()
        self._setup_transforms()

        # Строим индекс валидных примеров (без обработки картинок!)
        self._build_index()

    def _load_dataset(self, split):
        print(f"📥 Загрузка датасета (split={split})...")
        if self.test_size:
            full_ds = load_dataset("Leeps/Fonts-Individual-Letters", split="train")
            split_ds = full_ds.train_test_split(test_size=self.test_size, seed=self.seed)
            self.raw_data = split_ds[split]
        else:
            self.raw_data = load_dataset("Leeps/Fonts-Individual-Letters", split=split)
        print(f"   ✅ Загружено {len(self.raw_data)} примеров (в memory‑mapped)")

    def _create_alphabet(self):
        self.all_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:#'\"!?/()"
        self.char_to_idx = {c: i for i, c in enumerate(self.all_chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        print(f"🔤 Алфавит: {len(self.all_chars)} символов")

    def _setup_transforms(self):
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        print(f"🖼️  Трансформации: resize -> {self.img_size}x{self.img_size}, to tensor")

    def _clean_image(self, img):
        # Если изображение пришло как список/массив
        if isinstance(img, list):
            img = np.array(img, dtype=np.uint8)
            while img.ndim > 2:
                img = img[0]
            img = Image.fromarray(img)

        img = img.convert("L")
        # Инвертируем, если фон тёмный (буква светлая)
        if np.mean(np.array(img)) < 128:
            img = ImageOps.invert(img)
        return img

    def _extract_letter(self, text):
        if isinstance(text, list):
            text = text[0]
        match = re.search(r"TOK '([^']*)'", text)
        return match.group(1) if match else None

    def _build_index(self):
        """Проходим по всем сырым примерам, проверяем только текст и сохраняем индексы + метки"""
        print("🏗️ Построение индекса валидных примеров...")
        self.indices = []   # индексы в raw_data
        self.labels = []    # соответствующие метки
        for i, ex in enumerate(self.raw_data):
            letter = self._extract_letter(ex["text"])
            if letter and letter in self.char_to_idx:
                self.indices.append(i)
                self.labels.append(self.char_to_idx[letter])
            if (i + 1) % 10000 == 0:
                print(f"   ⏳ Обработано {i+1}/{len(self.raw_data)}...")
        print(f"   ✅ Найдено {len(self.indices)} валидных примеров")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        raw_idx = self.indices[idx]
        label = self.labels[idx]
        ex = self.raw_data[raw_idx]                # получаем сырой пример (из memory‑mapped)
        img = self._clean_image(ex["image"])       # чистим картинку
        img_tensor = self.transform(img)           # ресайзим и делаем тензор
        return {
            'image': img_tensor,
            'label': label,
            # 'raw_text': ex["text"] — если нужно для отладки, раскомментируй
        }

    def get_alphabet_size(self):
        return len(self.all_chars)

    def get_char_from_label(self, label):
        return self.idx_to_char.get(label, '?')

    def show_stats(self):
        print("\n📊 Статистика датасета:")
        print(f"   Всего примеров: {len(self)}")
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"   Классов: {len(unique)}")
        print(f"   Макс примеров в классе: {max(counts)}")
        print(f"   Мин примеров в классе: {min(counts)}")
        print(f"   Среднее: {np.mean(counts):.1f}")

    @staticmethod
    def collate_fn(batch):
        return {
            'image': torch.stack([b['image'] for b in batch]),
            'label': torch.tensor([b['label'] for b in batch], dtype=torch.long),
        }

def create_dataloaders(
    train_batch_size=32,
    test_batch_size=32,
    img_size=224,
    test_size=0.2,
    num_workers=2
):
    """
    Удобная функция для создания train/test даталоадеров
    
    Args:
        train_batch_size: размер батча для тренировки
        test_batch_size: размер батча для теста
        img_size: размер изображений
        test_size: доля тестовых данных (0.0-1.0)
        num_workers: количество процессов для загрузки
        
    Returns:
        tuple: (train_loader, test_loader, dataset_info)
    """
    print("🚀 Создание даталоадеров...")
    
    # Создаём train датасет
    train_dataset = FontLetterDataset(
        split="train",
        img_size=img_size,
        test_size=test_size
    )
    
    # Создаём test датасет
    test_dataset = FontLetterDataset(
        split="test",
        img_size=img_size,
        test_size=test_size
    )
    
    # Создаём даталоадеры
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=FontLetterDataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=FontLetterDataset.collate_fn
    )
    
    info = {
        'alphabet_size': train_dataset.get_alphabet_size(),
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'img_size': img_size
    }
    
    print(f"\n✅ Готово!")
    print(f"   Train: {info['train_size']} примеров")
    print(f"   Test:  {info['test_size']} примеров")
    print(f"   Классов: {info['alphabet_size']}")
    
    return train_loader, test_loader, info
