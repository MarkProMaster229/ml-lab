# modules/prompt_generator.py
import json
import random
from typing import Dict
from config import Config

class PromptGenerator:
    """Генератор промптов для модели с расширенным контекстом"""
    
    def __init__(self, config: Config):
        self.config = config
        
        self.system_prompt = """Ты - генератор синтетических данных для обучения ИИ. 
Твоя задача - создавать короткие (не более 30-45 слов) пары вопрос-ответ в строгом JSON формате.
Ты всегда возвращаешь ТОЛЬКО JSON объект без каких-либо пояснений, комментариев или форматирования.
Формат всегда такой: {"input": "вопрос", "target": "ответ"}"""
        
        self.examples = json.dumps([
            {
                "input": "Привет, как дела?",
                "target": "У меня всё хорошо, спасибо!"
            },
            {
                "input": "Какая сегодня погода?",
                "target": "Сегодня солнечно и тепло."
            },
            {
                "input": "Как настроить Wi-Fi роутер?",
                "target": "Подключите кабель к WAN порту, зайдите в настройки через 192.168.1.1."
            }
        ], ensure_ascii=False)
    
    def generate_prompt(self) -> str:
        """Генерирует случайный промпт с расширенным контекстом"""
        context = self.config.get_random_context()
        
        prompt = f"""{self.system_prompt}

КОНТЕКСТ ДИАЛОГА:
• Тема: {context['topic']}
• Сценарий: {context['scenario']}
• Отрасль: {context['industry']}
• Профессия участника: {context['profession']}
• Эмоциональный тон: {context['emotion']}
• Сложность: {context['difficulty']}
• Стиль общения: {context['style']}

Примеры правильного формата:
{self.examples}

Создай ОДИН уникальный диалог в заданном контексте. 
Верни ТОЛЬКО JSON объект, ничего больше:"""
        
        return prompt
    
    def generate_specialized_prompt(self, category: str = None) -> str:
        """Генерирует специализированный промпт"""
        if category == "technical":
            return self._generate_technical_prompt()
        elif category == "medical":
            return self._generate_medical_prompt()
        elif category == "business":
            return self._generate_business_prompt()
        elif category == "creative":
            return self._generate_creative_prompt()
        else:
            return self.generate_prompt()
    
    def _generate_technical_prompt(self) -> str:
        """Генерирует технический промпт"""
        tech_topics = [
            "программирование", "базы данных", "сети", "безопасность",
            "DevOps", "мобильная разработка", "веб-разработка", "AI/ML"
        ]
        
        context = {
            "topic": random.choice(tech_topics),
            "scenario": random.choice(["баг-репорт", "техподдержка", "код ревью", "архитектурное решение"]),
            "style": "технический"
        }
        
        return f"""{self.system_prompt}

ТЕХНИЧЕСКИЙ ДИАЛОГ:
Тема: {context['topic']}
Сценарий: {context['scenario']}
Стиль: {context['style']}

Примеры:
{self.examples}

Создай технический диалог. Верни ТОЛЬКО JSON:"""
    
    def generate_batch_prompts(self, count: int, mix_categories: bool = False) -> list[str]:
        """Генерирует несколько промптов с возможностью смешивания категорий"""
        if mix_categories:
            categories = [None, "technical", "medical", "business", "creative"]
            return [self.generate_specialized_prompt(random.choice(categories)) 
                   for _ in range(count)]
        else:
            return [self.generate_prompt() for _ in range(count)]