# config.py
import os
from dataclasses import dataclass
from typing import List, Dict, Any
import random

@dataclass
class Config:
    """Конфигурация генератора интернет-комментариев"""
    model_name: str = "ministral-3:latest"
    ollama_url: str = "http://localhost:11434/api/generate"
    target_count: int = 5000
    delay_between_requests: float = 1.5
    output_filename: str = "real_comments_dataset.json"
    
    # Настройки для интернет-комментариев
    platforms: List[str] = None
    content_topics: List[str] = None
    
    # Настройки модели
    temperature: float = 0.9
    top_p: float = 0.95
    num_predict: int = 200  # Уменьшил, т.к. комментарии короткие
    repeat_penalty: float = 1.1
    
    def __post_init__(self):
        if self.platforms is None:
            self.platforms = ["youtube", "tiktok", "twitch", "vk", "telegram"]
        
        if self.content_topics is None:
            self.content_topics = self._get_real_topics()
    
    def _get_real_topics(self) -> List[str]:
        """Реальные темы для видео"""
        return [
            # Самые популярные
            "обзор игры",
            "реакция на мем",
            "распаковка заказа с алиэкспресс",
            "летсплей майнкрафт/кс/дота",
            "тикток тренд",
            "мобильная игра",
            "стрим по игре",
            "юмористический скетч",
            "обзор айфона/андроида",
            "музыкальный клип",
            "обзор фильма/сериала",
            "косплей",
            "вызов/челлендж",
            "лайфхак",
            "новости игр",
            "прогулка по городу",
            "еда/рецепт",
            "спортивные моменты",
            "пранк/розыгрыш",
            "животные/коты",
            "машины/тюнинг",
            "строительство/ремонт",
            "путешествия",
            "спойлер фильма",
            "теория по игре/фильму",
            "баги/глюки в игре",
            "достижение/рекорд",
            "коллекционирование",
            "аниме/манга",
            "стример донатит",
            "фейл/неудача",
            "угарает с чего-то",
            "обзор наушников",
            "сборка пк",
            "крипота/страшилки"
        ]
    
    def get_random_platform(self) -> str:
        """Возвращает случайную платформу"""
        return random.choice(self.platforms)
    
    def get_random_topic(self) -> str:
        """Возвращает случайную тему"""
        return random.choice(self.content_topics)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Создает конфиг из словаря"""
        return cls(**config_dict)