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

    def generate_classification_prompt(self, comment: str) -> str:
        """Генерирует промпт для классификации одного комментария"""
        
        prompt = f"""{self.system_prompt}

АНАЛИЗИРУЙ КОММЕНТАРИЙ:
"{comment}"

ВОПРОС: Какую тональность имеет этот комментарий?

ОТВЕТЬ ТОЛЬКО В ФОРМАТЕ JSON:
{{"label": "positive"}} или {{"label": "negative"}} или {{"label": "neutral"}}"""
        
        return prompt