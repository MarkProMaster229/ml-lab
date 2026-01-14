# prompt_generator.py
import json
import random
from typing import Dict, List
from config import Config

class PromptGenerator:
    """Генератор промптов для классификации комментариев"""
    
    def __init__(self, config: Config):
        self.config = config
        
        self.system_prompt = """Ты - классификатор тональности комментариев из соцсетей.
Проанализируй комментарий и определи его тональность.

ПРАВИЛА КЛАССИФИКАЦИИ:
1. POSITIVE (позитивный) - комментарий выражает одобрение, поддержку, радость, благодарность
2. NEGATIVE (негативный) - комментарий выражает недовольство, критику, злость, раздражение
3. NEUTRAL (нейтральный) - комментарий информационный, вопрос, без явной эмоциональной окраски

Ты всегда возвращаешь ТОЛЬКО JSON объект с одним полем 'label'."""

        # Примеры для обучения модели
        self.examples = {
            "positive": [
                "Классное видео, спасибо автору!",
                "Супер, мне понравилось",
                "Очень полезно, беру на заметку",
                "Лучшее что я видел за последнее время",
                "Автор молодец, так держать"
            ],
            "negative": [
                "Полный бред, не советую смотреть",
                "Автор вообще в теме?",
                "Скучно и неинтересно",
                "Зря потратил время",
                "Это просто ужасно"
            ],
            "neutral": [
                "А можно подробнее про третий пункт?",
                "Сколько времени ушло на съемку?",
                "Как называется программа которую вы используете?",
                "Где можно найти ссылку на статью?",
                "Какая версия игры используется в видео?"
            ]
        }
    
    def generate_classification_prompt(self, comment: str) -> str:
        """Генерирует промпт для классификации одного комментария"""
        
        # Выбираем по 2 примера каждого типа для контекста
        pos_examples = random.sample(self.examples["positive"], 2)
        neg_examples = random.sample(self.examples["negative"], 2)
        neu_examples = random.sample(self.examples["neutral"], 2)
        
        prompt = f"""{self.system_prompt}

КОНТЕКСТНЫЕ ПРИМЕРЫ:
Позитивные (positive):
{chr(10).join(f'• "{example}"' for example in pos_examples)}

Негативные (negative):
{chr(10).join(f'• "{example}"' for example in neg_examples)}

Нейтральные (neutral):
{chr(10).join(f'• "{example}"' for example in neu_examples)}

АНАЛИЗИРУЙ КОММЕНТАРИЙ:
"{comment}"

ВОПРОС: Какую тональность имеет этот комментарий?

ОТВЕТЬ ТОЛЬКО В ФОРМАТЕ JSON:
{{"label": "positive"}} или {{"label": "negative"}} или {{"label": "neutral"}}"""
        
        return prompt