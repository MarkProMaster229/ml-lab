#!/usr/bin/env python3
"""
Split JSON dataset into train/validation.
Просто запусти: python split_dataset.py
"""

import json
import random
from pathlib import Path

# ==================== КОНФИГУРАЦИЯ ====================
INPUT_FILE = "/home/chelovek/Desktop/new/biginit.json"  # путь к датасету
OUTPUT_DIR = "/home/chelovek/Desktop/new/"    # куда сохранить
VAL_SIZE = 0.1                                               # доля валидации (10%)
SEED = 42                                                    # для воспроизводимости
SHUFFLE = True                                               # перемешивать ли
# ======================================================


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Ожидается JSON-массив объектов")
    return data


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    input_path = Path(INPUT_FILE)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🔪 НАРЕЗКА ДАТАСЕТА")
    print("=" * 60)
    print(f"📂 Вход: {input_path}")
    print(f"📂 Выход: {output_dir}")
    print(f"✂️ Val size: {VAL_SIZE}")
    print(f"🌱 Seed: {SEED}")
    print(f"🔀 Shuffle: {SHUFFLE}")

    # Загрузка
    print("\n📥 Загружаю...")
    data = load_json(input_path)
    total = len(data)
    print(f"📊 Всего примеров: {total}")

    # Перемешивание
    if SHUFFLE:
        rng = random.Random(SEED)
        indices = list(range(total))
        rng.shuffle(indices)
        data = [data[i] for i in indices]
        print("🔀 Перемешано")

    # Разделение
    val_count = int(total * VAL_SIZE)
    train_count = total - val_count

    train_data = data[:train_count]
    val_data = data[train_count:]

    print(f"📁 Train: {train_count}")
    print(f"📁 Validation: {val_count}")

    # Сохранение
    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"

    save_json(train_data, train_path)
    save_json(val_data, val_path)

    print(f"\n✅ Train: {train_path}")
    print(f"✅ Validation: {val_path}")
    print("\nГотово!")


if __name__ == "__main__":
    main()