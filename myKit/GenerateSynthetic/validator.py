# modules/validator.py
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any

class ResponseValidator:
    """Улучшенный валидатор ответов модели."""
    
    def __init__(self):
        self.meaningless_phrases = [
            "не знаю", "нет ответа", "не понимаю", "...", "???",
            "не могу", "извините", "я не уверен", "сложно сказать"
        ]
    
    def validate_batch(self, response_text: str) -> List[Dict]:
        """Парсит JSON-массив из ответа модели."""
        if not response_text:
            print("❌ Пустой ответ от модели")
            return []
        
        # Сохраняем сырой ответ для отладки
        self._log_raw_response(response_text)
        
        # Пробуем разные методы парсинга
        json_arrays = []
        
        # Метод 0: Прямой парсинг JSON (самый строгий)
        json_arrays.append(self._parse_direct_json(response_text))
        
        # Метод 1: Поиск массива в тексте с балансировкой
        json_arrays.append(self._find_json_array(response_text))
        
        # Метод 2: Извлечение отдельных объектов
        json_arrays.append(self._extract_individual_objects(response_text))
        
        # Метод 3: Парсинг Python-подобного списка
        json_arrays.append(self._parse_python_list(response_text))
        
        # Метод 4: Поиск JSON внутри code blocks
        json_arrays.append(self._extract_json_from_code_blocks(response_text))
        
        # Выбираем самый большой валидный массив
        best_array = []
        for array in json_arrays:
            if array and len(array) > len(best_array):
                if self._validate_array_structure(array):
                    best_array = array
        
        if not best_array:
            print("❌ Не удалось извлечь валидные данные ни одним методом")
            print(f"📝 Ответ модели (первые 1000 символов):")
            print(response_text[:1000] + ("..." if len(response_text) > 1000 else ""))
            return []
        
        # Валидация элементов
        valid_items = []
        for idx, item in enumerate(best_array, 1):
            if self._validate_item(item):
                valid_items.append(item)
            else:
                print(f"⚠️ Отклонён элемент #{idx}")
        
        print(f"✅ Принято {len(valid_items)} из {len(best_array)} элементов")
        
        if not valid_items:
            print("⚠️ Ни один элемент не прошел валидацию")
            print("Попробуйте изменить промпт или снизить strictness валидации")
        
        return valid_items
    
    def _log_raw_response(self, response: str):
        """Сохраняет сырой ответ для отладки."""
        try:
            with open("raw_responses.log", "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Длина: {len(response)} символов\n")
                f.write(f"Содержимое:\n{response}\n")
                f.write(f"{'='*80}\n")
        except Exception as e:
            print(f"⚠️ Ошибка при логировании: {e}")
    
    def _parse_direct_json(self, text: str) -> List[Dict]:
        """Прямой парсинг JSON."""
        text = text.strip()
        if not text:
            return []
        
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Пробуем найти массив в словаре
                for value in data.values():
                    if isinstance(value, list):
                        return value
        except json.JSONDecodeError as e:
            pass
        return []
    
    def _find_json_array(self, text: str) -> List[Dict]:
        """Поиск JSON массива в тексте с балансировкой скобок."""
        # Ищем начало массива
        start = text.find('[')
        if start == -1:
            return []
        
        # Находим соответствующий конец массива
        brackets = 0
        in_string = False
        escape = False
        
        for i in range(start, len(text)):
            char = text[i]
            
            if escape:
                escape = False
                continue
                
            if char == '\\':
                escape = True
                continue
                
            if char == '"' and not escape:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '[':
                    brackets += 1
                elif char == ']':
                    brackets -= 1
                    if brackets == 0:
                        # Найден конец массива
                        try:
                            json_str = text[start:i+1]
                            data = json.loads(json_str)
                            if isinstance(data, list):
                                return data
                        except json.JSONDecodeError:
                            # Пробуем починить JSON
                            json_str = self._repair_json(json_str)
                            if json_str:
                                try:
                                    data = json.loads(json_str)
                                    if isinstance(data, list):
                                        return data
                                except:
                                    pass
                        break
        
        return []
    
    def _repair_json(self, json_str: str) -> str:
        """Пытается починить сломанный JSON."""
        if not json_str:
            return ""
        
        repairs = []
        
        # 1. Убираем лишние запятые перед закрывающими скобками
        json_str = re.sub(r',\s*\]', ']', json_str)
        json_str = re.sub(r',\s*\}', '}', json_str)
        
        # 2. Исправляем незакрытые строки
        lines = json_str.split('\n')
        fixed_lines = []
        in_string = False
        escape = False
        
        for line in lines:
            new_line = ""
            for char in line:
                if escape:
                    escape = False
                    new_line += char
                    continue
                    
                if char == '\\':
                    escape = True
                    new_line += char
                    continue
                    
                if char == '"':
                    in_string = not in_string
                
                new_line += char
            
            fixed_lines.append(new_line)
        
        json_str = '\n'.join(fixed_lines)
        
        # 3. Убираем непечатаемые символы, кроме разрешенных
        allowed_chars = set(' \t\n\r')
        json_str = ''.join(char for char in json_str if char.isprintable() or char in allowed_chars)
        
        return json_str
    
    def _extract_json_from_code_blocks(self, text: str) -> List[Dict]:
        """Извлекает JSON из code blocks (```json ... ```)."""
        pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(pattern, text)
        
        for match in matches:
            # Пробуем распарсить как JSON
            result = self._parse_direct_json(match)
            if result:
                return result
        
        return []
    
    def _extract_individual_objects(self, text: str) -> List[Dict]:
        """Извлекает отдельные JSON объекты из текста."""
        objects = []
        
        # Упрощаем текст для поиска
        simplified = re.sub(r'\s+', ' ', text)
        
        # Паттерн для поиска объектов в формате JSON
        # Более точный паттерн для объектов с input и target
        pattern = r'\{\s*["\']?input["\']?\s*:\s*["\']([^"\']*)["\'][^}]*["\']?target["\']?\s*:\s*["\']([^"\']*)["\'][^}]*\}'
        
        matches = re.finditer(pattern, simplified, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            try:
                # Создаем объект из найденных групп
                input_text = match.group(1).strip()
                target_text = match.group(2).strip()
                
                if input_text and target_text:
                    objects.append({
                        "input": input_text,
                        "target": target_text
                    })
            except Exception as e:
                continue
        
        return objects
    
    def _parse_python_list(self, text: str) -> List[Dict]:
        """Парсит Python-подобный список."""
        objects = []
        
        # Паттерн для Python-словарей
        pattern = r'\{\s*["\']?input["\']?\s*:\s*["\']([^"\']*)["\'][^}]*["\']?target["\']?\s*:\s*["\']([^"\']*)["\'][^}]*\}'
        
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            try:
                input_text = match.group(1).strip()
                target_text = match.group(2).strip()
                
                if input_text and target_text:
                    objects.append({
                        "input": input_text,
                        "target": target_text
                    })
            except Exception as e:
                continue
        
        return objects
    
    def _validate_array_structure(self, array: List) -> bool:
        """Проверяет базовую структуру массива."""
        if not isinstance(array, list):
            return False
        
        if not array:
            return False
        
        # Проверяем первые 3 элемента
        for i, item in enumerate(array[:3]):
            if not isinstance(item, dict):
                return False
            
            if "input" not in item or "target" not in item:
                return False
        
        return True
    
    def _validate_item(self, item: Dict) -> bool:
        """Валидирует отдельный элемент."""
        # Базовые проверки
        if not isinstance(item, dict):
            return False
        
        if "input" not in item or "target" not in item:
            return False
        
        input_text = str(item["input"]).strip()
        target_text = str(item["target"]).strip()
        
        # Проверка на пустые строки
        if not input_text or not target_text:
            return False
        
        # Минимальная длина
        if len(input_text) < 5 or len(target_text) < 5:
            return False
        
        # Проверка на бессмысленные ответы
        target_lower = target_text.lower()
        if any(phrase in target_lower for phrase in self.meaningless_phrases):
            return False
        
        # Проверка на слишком похожие строки
        input_words = set(re.findall(r'\w+', input_text.lower()))
        target_words = set(re.findall(r'\w+', target_text.lower()))
        
        if input_words and target_words:
            intersection = len(input_words.intersection(target_words))
            similarity = intersection / max(len(input_words), 1)
            if similarity > 0.8:  # 80% совпадение
                return False
        
        # Максимальная длина
        if len(input_text) > 10000 or len(target_text) > 10000:
            return False
        
        return True