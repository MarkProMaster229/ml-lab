# config.py
import os
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    """Конфигурация генератора"""
    model_name: str = "mixtral:latest"
    ollama_url: str = "http://localhost:11434/api/generate"
    target_count: int = 5000
    delay_between_requests: float = 2.0
    output_filename: str = "synthetic_dataset.json"
    
    # Настройки промптов
    topics: List[str] = None
    scenarios: List[str] = None
    
    # Настройки модели
    temperature: float = 0.8
    top_p: float = 0.95
    num_predict: int = 350
    repeat_penalty: float = 1.2
    
    def __post_init__(self):
        if self.topics is None:
            self.topics = [
                "бытовой диалог", "техническая консультация", "покупка в магазине",
                "запрос в поддержку", "образовательный вопрос", "медицинская консультация",
                "путешествия", "кулинария", "финансы", "хобби и развлечения",
                "спорт и здоровье", "технологии", "искусство и культура", "работа и карьера",
                "отношения и общение", "наука и исследования", "природа и экология"
            ]
        
        if self.scenarios is None:
            self.scenarios = [
                "пользователь спрашивает, ассистент отвечает",
                "клиент жалуется, поддержка решает проблему",
                "студент задаёт вопрос, преподаватель объясняет",
                "покупатель уточняет детали, продавец консультирует",
                "друг советуется, друг даёт рекомендацию",
                "коллега просит помощи, коллега помогает",
                "посетитель спрашивает информацию, гид рассказывает"
            ]
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Создает конфиг из словаря"""
        return cls(**config_dict)