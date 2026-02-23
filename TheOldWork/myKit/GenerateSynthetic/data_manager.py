# modules/data_manager.py
import json
import os
import hashlib
from typing import List, Dict

class DataManager:
    """Простой менеджер данных."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.seen_hashes = set()
        self._load_existing()
    
    def _load_existing(self):
        """Загружает существующие данные и их хеши."""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        text = f"{item.get('input', '')}{item.get('target', '')}"
                        self.seen_hashes.add(hashlib.md5(text.encode()).hexdigest())
            except:
                pass
    
    def add_example(self, example: Dict) -> bool:
        """Добавляет пример, проверяя дубликаты."""
        # Проверка дубликата
        text = f"{example.get('input', '')}{example.get('target', '')}"
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.seen_hashes:
            print("⚠️ Дубликат, пропускаю")
            return False
        
        # Загрузка и добавление
        data = []
        if os.path.exists(self.filename):
            with open(self.filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        data.append(example)
        
        # Сохранение
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.seen_hashes.add(text_hash)
        print(f"💾 Сохранено! Всего: {len(data)}")
        return True