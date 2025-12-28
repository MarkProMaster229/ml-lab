# modules/prompt_generator.py
import json
import random
from typing import Dict
from config import Config

class PromptGenerator:
    """Генератор промптов для модели"""
    
    def __init__(self, config: Config):
        self.config = config
        self.styles = ["формальный", "неформальный", "технический", "простой"]
        
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
                "input": "Как приготовить пасту?",
                "target": "Нужно отварить спагетти и добавить соус."
            }
        ], ensure_ascii=False)
    
    def generate_prompt(self) -> str:
        """Генерирует случайный промпт"""
        topic = random.choice(self.config.topics)
        scenario = random.choice(self.config.scenarios)
        style = random.choice(self.styles)
        
        prompt = f"""{self.system_prompt}

ТЕМА: {topic}
СЦЕНАРИЙ: {scenario}
СТИЛЬ: {style}

Примеры правильного формата:
{self.examples}

Создай ОДИН уникальный диалог на заданную тему. 
Верни ТОЛЬКО JSON объект, ничего больше:"""
        
        return prompt
    
    def generate_batch_prompts(self, count: int) -> list[str]:
        """Генерирует несколько промптов"""
        return [self.generate_prompt() for _ in range(count)]