# data_manager.py
import json
import os
from datetime import datetime
from typing import Dict, List

class DataManager:
    """Менеджер данных: загрузка, классификация, сохранение"""
    
    def __init__(self, input_filename: str, output_filename: str):
        self.input_filename = input_filename
        self.output_filename = output_filename
        
    def load_input_data(self) -> List[Dict]:
        """Загружает данные для классификации"""
        if not os.path.exists(self.input_filename):
            print(f"⚠️ Входной файл {self.input_filename} не найден")
            return []
        
        try:
            with open(self.input_filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    # Проверяем формат входных данных
                    validated_data = []
                    for i, item in enumerate(data):
                        if isinstance(item, dict) and 'text' in item:
                            validated_data.append({'text': item['text']})
                        elif isinstance(item, str):
                            validated_data.append({'text': item})
                        else:
                            print(f"⚠️ Пропускаю элемент {i}: неподдерживаемый формат")
                    return validated_data
                else:
                    print(f"⚠️ Файл содержит не список, а {type(data)}")
                    return []
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Ошибка загрузки входного файла: {e}")
            return []
    
    def load_existing_classified_data(self) -> List[Dict]:
        """Загружает уже классифицированные данные"""
        if not os.path.exists(self.output_filename):
            return []
        
        try:
            with open(self.output_filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    print(f"⚠️ Выходной файл содержит не список, а {type(data)}")
                    return []
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Ошибка загрузки выходного файла: {e}")
            return []
    
    def save_classified_data(self, data: List[Dict]) -> bool:
        """Сохраняет классифицированные данные"""
        try:
            # Сохраняем в красивом формате
            with open(self.output_filename, 'w', encoding='utf-8') as f:
                f.write('[\n')
                for i, item in enumerate(data):
                    json_str = json.dumps(item, ensure_ascii=False, indent=2)
                    if i < len(data) - 1:
                        f.write(f'  {json_str.replace(chr(10), chr(10) + "  ")},\n')
                    else:
                        f.write(f'  {json_str.replace(chr(10), chr(10) + "  ")}\n')
                f.write(']')
            
            print(f"💾 Сохранено {len(data)} записей в {self.output_filename}")
            return True
                
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
            return False
    
    def add_classified_comment(self, comment: str, label: str) -> bool:
        """Добавляет один классифицированный комментарий в файл"""
        existing_data = self.load_existing_classified_data()
        
        # Проверка на дубликаты
        for existing in existing_data:
            if existing.get("text", "") == comment:
                print(f"⚠️ Комментарий уже существует, пропускаю")
                return False
        
        new_entry = {
            "text": comment,
            "label": label
        }
        
        existing_data.append(new_entry)
        return self.save_classified_data(existing_data)
    
    def get_unclassified_comments(self, input_data: List[Dict], classified_data: List[Dict]) -> List[str]:
        """Возвращает список еще не классифицированных комментариев"""
        classified_texts = {item['text'] for item in classified_data if 'text' in item}
        unclassified = []
        
        for item in input_data:
            text = item['text']
            if text not in classified_texts:
                unclassified.append(text)
        
        return unclassified