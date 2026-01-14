# config.py
import os
from dataclasses import dataclass
from typing import List, Dict, Any
import random

@dataclass
class Config:
    """Конфигурация классификатора комментариев"""
    model_name: str = "ministral-3:latest"
    ollama_url: str = "http://localhost:11434/api/generate"
    delay_between_requests: float = 1.5
    
    # Настройки для классификации
    input_filename: str = "comments_to_classify.json"
    output_filename: str = "classified_comments.json"
    
    # Настройки модели
    temperature: float = 0.3  # Понизим для более консистентной классификации
    top_p: float = 0.9
    num_predict: int = 50  # Нужно мало токенов, только метку
    repeat_penalty: float = 1.1
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Создает конфиг из словаря"""
        return cls(**config_dict)