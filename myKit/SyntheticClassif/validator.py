# modules/validator.py
import json
import re
from typing import Dict, Optional

class ResponseValidator:
    """Валидатор ответов модели для классификационного формата"""
    
    def __init__(self):
        self.meaningless_phrases = [
            "не знаю", "нет ответа", "не понимаю", "...", "???", 
            "не могу", "извините", "я не уверен", "сложно сказать",
            "как пример", "например", "вот текст", "сгенерирован",
            "это пример", "создаю текст", "вот сообщение"
        ]
        self.valid_labels = ["positive", "negative", "neutral"]
    
    def validate(self, response_text: str, required_label: str) -> Optional[Dict]:
        """Проверяет и валидирует ответ модели"""
        if not response_text:
            print("❌ Пустой ответ")
            return None
        
        response_text = response_text.strip()
        print(f"📨 Сырой ответ ({len(response_text)} chars): {response_text[:150]}...")
        
        # Поиск JSON
        data = self._find_and_parse_json(response_text)
        if data and self._validate_classification_format(data, required_label):
            print(f"✅ Валидный JSON: текст({len(data.get('text', ''))} chars), label: {data.get('label', 'N/A')}")
            return data
        
        print("❌ Невалидный JSON или формат")
        return None
    
    def _find_and_parse_json(self, text: str) -> Optional[Dict]:
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
        data = self._try_parse_json(text)
        if data:
            return data
        
        # Поиск в тексте (более глубокий поиск)
        # Ищем любые JSON структуры
        json_candidates = []
        
        # Поиск с помощью регулярных выражений
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(pattern, text, re.DOTALL)
        
        for match in matches:
            candidate = match.group(0)
            if candidate.count('{') == candidate.count('}'):
                json_candidates.append(candidate)
        
        for candidate in json_candidates:
            data = self._try_parse_json(candidate)
            if data:
                print(f"✅ Найден JSON в тексте: {candidate[:100]}...")
                return data
        
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
            
            # Проверка структуры для классификационного формата
            if not isinstance(data, dict):
                return None
            
            # Должен быть текст и метка
            if "text" not in data or "label" not in data:
                return None
            
            # Проверяем метку
            if data["label"] not in self.valid_labels:
                return None
            
            # Очистка
            data["text"] = data["text"].strip()
            data["label"] = data["label"].strip().lower()
            
            return data
            
        except json.JSONDecodeError:
            return None
        except Exception as e:
            print(f"⚠️ Ошибка парсинга JSON: {e}")
            return None
    
    def _validate_classification_format(self, data: Dict, required_label: str) -> bool:
        """Дополнительные проверки качества для классификационного формата"""
        # Проверка текста
        text = data.get("text", "")
        label = data.get("label", "")
        
        if not text:
            print("❌ Пустой текст")
            return False
        
        # Проверка метки
        if label != required_label:
            print(f"❌ Метка не совпадает: ожидалось '{required_label}', получено '{label}'")
            return False
        
        # Слишком короткие
        word_count = len(text.split())
        if word_count < 15:
            print(f"⚠️ Слишком короткий текст: {word_count} слов (минимум 15)")
            return False
        
        # Слишком длинные (более 200 слов)
        if word_count > 200:
            print(f"⚠️ Слишком длинный текст: {word_count} слов (максимум 200)")
            return False
        
        # Бессмысленные тексты
        if any(phrase in text.lower() for phrase in self.meaningless_phrases):
            print("❌ Текст содержит бессмысленные фразы")
            return False
        
        # Проверка соответствия текста метке
        if not self._check_label_consistency(text, label):
            print(f"⚠️ Текст не соответствует метке '{label}'")
            return False
        
        return True
    
    def _check_label_consistency(self, text: str, label: str) -> bool:
        """Проверяет, соответствует ли текст метке"""
        text_lower = text.lower()
        
        if label == "positive":
            positive_words = ["отличн", "хорош", "спасибо", "рекоменд", "довол", "рад", "супер", "замечательн", "прекрасн"]
            return any(word in text_lower for word in positive_words)
        
        elif label == "negative":
            negative_words = ["плох", "ужас", "кошмар", "жалоб", "недовол", "отврат", "разочарован", "груб", "грязн", "сломал"]
            return any(word in text_lower for word in negative_words)
        
        else:  # neutral
            # Нейтральный текст обычно содержит вопросы или информацию
            question_words = ["подскажит", "уточнит", "интересует", "вопрос", "сколько", "когда", "где", "можно ли"]
            info_words = ["информац", "сообщаю", "уведомляю", "подтвержден", "заказ", "документ"]
            return any(word in text_lower for word in question_words + info_words)