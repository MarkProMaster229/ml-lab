# modules/data_manager.py
import json
import os
from datetime import datetime
from typing import Dict, List

class DataManager:
    """Менеджер данных: сохранение, загрузка, backup"""
    
    def __init__(self, output_filename: str):
        self.output_filename = output_filename
        self.backup_dir = "backups"
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def load_existing_data(self) -> List[Dict]:
        """Загружает существующие данные"""
        if not os.path.exists(self.output_filename):
            return []
        
        try:
            with open(self.output_filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    print(f"⚠️ Файл содержит не список, а {type(data)}")
                    return []
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Ошибка загрузки файла, создаём новый: {e}")
            return []
    
    def save_data(self, data: List[Dict]) -> bool:
        """Сохраняет данные в формате JSON"""
        try:
            # Основной файл с красивым форматированием
            with open(self.output_filename, 'w', encoding='utf-8') as f:
                f.write('[\n')
                for i, item in enumerate(data):
                    json_str = json.dumps(item, ensure_ascii=False, indent=4)
                    if i < len(data) - 1:
                        f.write(f'    {json_str.replace(chr(10), chr(10) + "    ")},\n')
                    else:
                        f.write(f'    {json_str.replace(chr(10), chr(10) + "    ")}\n')
                f.write(']')
            
            # Компактная копия
            compact_file = self.output_filename.replace('.json', '_compact.json')
            with open(compact_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            
            return True
                
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
            return False
    
    def add_example(self, example: Dict) -> bool:
        """Добавляет один пример в файл"""
        existing_data = self.load_existing_data()
        
        # Проверка на дубликаты (последние 100)
        for existing in existing_data[-100:]:
            if existing["input"] == example["input"] or existing["target"] == example["target"]:
                print(f"⚠️ Возможный дубликат, пропускаю")
                return False
        
        existing_data.append(example)
        
        if self.save_data(existing_data):
            print(f"💾 Сохранено! Всего записей: {len(existing_data)}")
            
            # Периодический backup
            if len(existing_data) % 100 == 0:
                self.create_backup(existing_data)
            
            return True
        
        return False
    
    def create_backup(self, data: List[Dict]) -> str:
        """Создает backup файл"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(self.backup_dir, f"{self.output_filename}_{timestamp}_backup.json")
        
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write('[\n')
                for i, item in enumerate(data):
                    json_str = json.dumps(item, ensure_ascii=False, indent=4)
                    if i < len(data) - 1:
                        f.write(f'    {json_str.replace(chr(10), chr(10) + "    ")},\n')
                    else:
                        f.write(f'    {json_str.replace(chr(10), chr(10) + "    ")}\n')
                f.write(']')
            
            print(f"📦 Backup создан: {backup_file}")
            return backup_file
            
        except Exception as e:
            print(f"❌ Ошибка создания backup: {e}")
            return ""
    
    def emergency_save(self, example: Dict) -> str:
        """Экстренное сохранение"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        emergency_file = f"emergency_save_{timestamp}.json"
        
        try:
            with open(emergency_file, 'w', encoding='utf-8') as f:
                f.write('[\n')
                json_str = json.dumps(example, ensure_ascii=False, indent=4)
                f.write(f'    {json_str.replace(chr(10), chr(10) + "    ")}\n')
                f.write(']')
            
            print(f"⚠️ Экстренно сохранено в {emergency_file}")
            return emergency_file
            
        except Exception:
            print("💥 Критическая ошибка сохранения!")
            return ""