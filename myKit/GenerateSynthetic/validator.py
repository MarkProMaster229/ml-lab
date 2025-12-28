# modules/validator.py
import json
import re
from typing import Dict, Optional

class ResponseValidator:
    """Валидатор ответов модели"""
    
    def __init__(self):
        self.meaningless_phrases = [
            "не знаю", "нет ответа", "не понимаю", "...", "???", 
            "не могу", "извините", "я не уверен", "сложно сказать"
        ]
    
    def validate(self, response_text: str) -> Optional[Dict]:
        """Проверяет и валидирует ответ модели"""
        if not response_text:
            print("❌ Пустой ответ")
            return None
        
        response_text = response_text.strip()
        print(f"📨 Сырой ответ ({len(response_text)} chars): {response_text[:150]}...")
        
        # Поиск JSON
        data = self._find_and_parse_json(response_text)
        if data and self._advanced_validation(data):
            print(f"✅ Валидный JSON: '{data['input'][:40]}...' → '{data['target'][:40]}...'")
            return data
        
        return None
    
    def _find_and_parse_json(self, text: str) -> Optional[Dict]:
        """Ищет и парсит JSON в тексте"""
        # Очистка markdown
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        # Прямой парсинг
        data = self._try_parse_json(text)
        if data:
            return data
        
        # Поиск в тексте
        json_candidates = re.findall(r'\{[^{}]*\}', text)
        for candidate in json_candidates:
            data = self._try_parse_json(candidate)
            if data:
                print(f"✅ Найден JSON в тексте: {candidate[:80]}...")
                return data
        
        print("❌ Никаких JSON структур не найдено")
        return None
    
    def _try_parse_json(self, text: str) -> Optional[Dict]:
        """Пробует распарсить текст как JSON"""
        if not text or not text.strip():
            return None
        
        text = text.strip()
        if not text.startswith('{') or not text.endswith('}'):
            return None
        
        try:
            data = json.loads(text)
            
            # Базовая проверка структуры
            if not isinstance(data, dict):
                return None
            if "input" not in data or "target" not in data:
                return None
            if not isinstance(data["input"], str) or not isinstance(data["target"], str):
                return None
            
            # Очистка
            data["input"] = data["input"].strip()
            data["target"] = data["target"].strip()
            
            # Проверка длины
            if len(data["input"]) < 3 or len(data["target"]) < 3:
                return None
            
            return data
            
        except json.JSONDecodeError:
            return None
        except Exception:
            return None
    
    def _advanced_validation(self, data: Dict) -> bool:
        """Дополнительные проверки качества"""
        # Пустые строки
        if not data["input"] or not data["target"]:
            return False
        
        # Слишком короткие
        if len(data["input"].split()) < 2 or len(data["target"].split()) < 2:
            return False
        
        # Повторение слов
        input_words = set(data["input"].lower().split())
        target_words = set(data["target"].lower().split())
        if len(input_words.intersection(target_words)) / max(len(input_words), 1) > 0.8:
            return False
        
        # Бессмысленные ответы
        if any(phrase in data["target"].lower() for phrase in self.meaningless_phrases):
            return False
        
        # Одинаковое начало
        if data["input"][:20] == data["target"][:20]:
            return False
        
        # Слишком длинные (более 45 слов)
        if len(data["input"].split()) > 75 or len(data["target"].split()) > 75:
            print(f"⚠️ Слишком длинный текст (более 75 слов)")
            return False
        
        return True