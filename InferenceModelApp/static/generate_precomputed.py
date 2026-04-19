#!/usr/bin/env python
"""
Генератор всех визуализаций для эмболии.
Запускается один раз, сохраняет все PNG в static/precomputed/
"""

from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Импортируем твой Manager
from main import Manager  # Замени на реальное имя файла

def main():
    # Пути к данным
    dataset_path = Path("/home/chelovek/Рабочий стол/PATFully/")  # Укажи свой путь
    dcm_dir = dataset_path / "CT_scans" / "PAT034"
    mat_path = dataset_path / "GroundTruth" / "PAT034.mat"
    output_root = Path("static/precomputed/PAT034")
    
    # Проверяем, что данные существуют
    if not dcm_dir.exists():
        print(f"❌ Ошибка: DICOM папка не найдена: {dcm_dir}")
        return
    
    if not mat_path.exists():
        print(f"⚠️ Предупреждение: GT файл не найден: {mat_path}")
        print("   Будет сохранено только КТ и предсказания")
    
    # Создаём менеджера
    manager = Manager()
    
    # Список моделей для экспорта
    models = [
        (1, "2D U-Net"),
        (2, "2D Attention U-Net"),
        (3, "3D U-Net"),
        (4, "3D Attention U-Net"),
    ]
    
    print("=" * 60)
    print("🚀 НАЧАЛО ГЕНЕРАЦИИ ВИЗУАЛИЗАЦИЙ")
    print("=" * 60)
    
    for model_id, model_name in models:
        print(f"\n📦 Обработка: {model_name}")
        manager.export_embol_visualizations(
            MyMagicObject=model_id,
            dcm_dir=dcm_dir,
            mat_path=mat_path,
            output_root=output_root
        )
    
    print("\n" + "=" * 60)
    print(f"✅ ГЕНЕРАЦИЯ ЗАВЕРШЕНА!")
    print(f"📁 Результаты сохранены в: {output_root.absolute()}")
    print("=" * 60)

if __name__ == "__main__":
    main()