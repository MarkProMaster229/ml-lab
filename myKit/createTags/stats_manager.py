# stats_manager.py
import time
from datetime import datetime
from typing import Dict

class StatsManager:
    """Менеджер статистики"""
    
    def __init__(self):
        self.stats = {
            "generated": 0,
            "failed": 0,
            "start_time": None,
            "consecutive_errors": 0
        }
    
    def start(self):
        """Начинает отсчёт статистики"""
        self.stats["start_time"] = time.time()
        self.stats["generated"] = 0
        self.stats["failed"] = 0
        self.stats["consecutive_errors"] = 0
    
    def add_success(self):
        """Добавляет успешную классификацию"""
        self.stats["generated"] += 1
        self.stats["consecutive_errors"] = 0
    
    def add_failure(self):
        """Добавляет неудачную классификацию"""
        self.stats["failed"] += 1
        self.stats["consecutive_errors"] += 1
    
    def get_stats(self) -> Dict:
        """Возвращает текущую статистику"""
        stats = self.stats.copy()
        
        if stats["start_time"]:
            elapsed = time.time() - stats["start_time"]
            stats["elapsed_seconds"] = elapsed
            stats["speed_per_minute"] = stats["generated"] / (elapsed / 60) if elapsed > 0 else 0
            
            total_attempts = stats["generated"] + stats["failed"]
            stats["success_rate"] = (stats["generated"] / total_attempts * 100) if total_attempts > 0 else 0
        
        return stats
    
    def print_stats(self):
        """Выводит статистику"""
        stats = self.get_stats()
        
        if "elapsed_seconds" in stats:
            print(f"\n{'='*60}")
            print(f"📊 СТАТИСТИКА:")
            print(f"   Успешно:     {stats['generated']}")
            print(f"   Ошибок:      {stats['failed']}")
            print(f"   Успешность:  {stats['success_rate']:.1f}%")
            print(f"   Время:       {stats['elapsed_seconds']:.0f} секунд")
            print(f"   Скорость:    {stats['speed_per_minute']:.1f} примеров/минуту")
            print(f"   Послед. ошибок: {stats['consecutive_errors']}")
            print(f"{'='*60}\n")