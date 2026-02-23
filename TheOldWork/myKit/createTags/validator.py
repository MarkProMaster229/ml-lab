# validator.py
import json
import re
from typing import Dict, Optional, List

class ResponseValidator:
    """Валидатор ответов модели для классификации"""
    
    def __init__(self):
        self.valid_labels = ["positive", "negative", "neutral"]
    
    def validate_classification(self, response_text: str) -> Optional[str]:
        """Проверяет и валидирует ответ модели с одной меткой"""
        if not response_text:
            print("❌ Пустой ответ")
            return None
        
        response_text = response_text.strip()
        print(f"📨 Сырой ответ ({len(response_text)} chars): {response_text[:150]}...")
        
        # Поиск JSON с меткой
        data = self._find_and_parse_json(response_text)
        if data and 'label' in data:
            label = data['label'].lower().strip()
            if label in self.valid_labels:
                print(f"✅ Валидная метка: {label}")
                return label
        
        # Если не нашли JSON, попробуем найти метку в тексте
        label = self._extract_label_from_text(response_text)
        if label:
            print(f"✅ Извлечена метка из текста: {label}")
            return label
        
        print("❌ Не удалось извлечь валидную метку")
        return None
    
    def _find_and_parse_json(self, text: str):
        """Ищет и парсит JSON в тексте"""
        # Очистка markdown
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        # Прямой парсинг
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Поиск в тексте
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(pattern, text, re.DOTALL)
        
        for match in matches:
            candidate = match.group(0)
            if candidate.count('{') == candidate.count('}'):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _extract_label_from_text(self, text: str) -> Optional[str]:
        """Пытается извлечь метку из текста"""
        text_lower = text.lower()
        
        # Ищем упоминания меток
        for label in self.valid_labels:
            if label in text_lower:
                # Проверяем контекст - метка должна быть отдельным словом
                pattern = r'\b' + label + r'\b'
                if re.search(pattern, text_lower):
                    return label
        
        # Ищем русские варианты
        russian_labels = {
            "позитив": "positive",
            "негатив": "negative", 
            "нейтрал": "neutral",
            "положительн": "positive",
            "отрицательн": "negative"
        }
        
        for russian, english in russian_labels.items():
            if russian in text_lower:
                return english
        
        return None