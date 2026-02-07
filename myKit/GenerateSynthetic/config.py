# modules/config.py
from dataclasses import dataclass, field
from typing import List, Dict
import random
import os

@dataclass
class Config:
    """Упрощенная конфигурация генератора."""
    
    # Основные параметры
    model_name: str = "devstral-2:123b-cloud"
    ollama_url: str = "http://localhost:11434/api/generate"
    target_count: int = 5000
    delay_between_requests: float = 2.0
    output_filename: str = "synthetic_dataset.json"
    
    # Параметры модели
    temperature: float = 0.8
    top_p: float = 0.95
    num_predict: int = 2000
    repeat_penalty: float = 1.2
    
    # Файлы с темами
    topics_file: str = "topics.txt"
    scenarios_file: str = "data/scenarios.txt"
    industries_file: str = "data/industries.txt"
    professions_file: str = "data/professions.txt"
    
    # Списки (будут загружены из файлов)
    topics: List[str] = field(default_factory=list)
    scenarios: List[str] = field(default_factory=list)
    industries: List[str] = field(default_factory=list)
    professions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Загружает данные из файлов."""
        self.topics = self._load_from_file(self.topics_file, self._default_topics())
        self.scenarios = self._load_from_file(self.scenarios_file, self._default_scenarios())
        self.industries = self._load_from_file(self.industries_file, self._default_industries())
        self.professions = self._load_from_file(self.professions_file, self._default_professions())
    
    @staticmethod
    def _load_from_file(filename: str, default: List[str]) -> List[str]:
        """Загружает строки из файла."""
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    print(f"📁 Загружено {len(lines)} строк из {filename}")
                    return lines
            except Exception as e:
                print(f"⚠️ Ошибка загрузки {filename}: {e}")
        return default
    
    # Дефолтные значения (если файлы не найдены)
    @staticmethod
    def _default_topics() -> List[str]:
        return ["разговор о природе", "обсуждение технологий", "личные переживания"]
    
    @staticmethod
    def _default_scenarios() -> List[str]:
        return ["утренняя беседа", "вечерний разговор", "случайная встреча"]
    
    @staticmethod
    def _default_industries() -> List[str]:
        return ["технологии", "образование", "развлечения"]
    
    @staticmethod
    def _default_professions() -> List[str]:
        return ["программист", "учитель", "художник"]
    
    def get_random_context(self) -> Dict[str, str]:
        """Возвращает случайный контекст для промпта."""
        return {
            "topic": random.choice(self.topics),
            "scenario": random.choice(self.scenarios),
            "industry": random.choice(self.industries),
            "profession": random.choice(self.professions),
        }