# modules/stats_manager.py
# -*- coding: utf-8 -*-

import time
from typing import Dict


class StatsManager:
    """Менеджер статистики генерации."""

    def __init__(self) -> None:
        self.stats = {
            "generated": 0,
            "failed": 0,
            "start_time": None,
            "context_resets": 0,
            "consecutive_errors": 0,
        }

    # ----------------------------------------------------------------- #
    def start(self) -> None:
        """Запускает таймер и обнуляет счётчики."""
        self.stats["start_time"] = time.time()
        self.stats["generated"] = 0
        self.stats["failed"] = 0
        self.stats["context_resets"] = 0
        self.stats["consecutive_errors"] = 0

    # ----------------------------------------------------------------- #
    def add_success(self, count: int = 1) -> None:
        """Увеличивает счётчик успешно сохранённых диалогов."""
        self.stats["generated"] += count
        self.stats["consecutive_errors"] = 0

    # ----------------------------------------------------------------- #
    def add_failure(self, count: int = 1) -> None:
        """Увеличивает счётчик ошибок."""
        self.stats["failed"] += count
        self.stats["consecutive_errors"] += count

    # ----------------------------------------------------------------- #
    def add_context_reset(self) -> None:
        self.stats["context_resets"] += 1

    # ----------------------------------------------------------------- #
    def get_stats(self) -> Dict:
        """Возвращает копию текущей статистики с вычисленными полями."""
        s = self.stats.copy()
        if s["start_time"]:
            elapsed = time.time() - s["start_time"]
            s["elapsed_seconds"] = elapsed
            s["speed_per_minute"] = s["generated"] / (elapsed / 60) if elapsed > 0 else 0
        total = s["generated"] + s["failed"]
        s["success_rate"] = (s["generated"] / total * 100) if total > 0 else 0
        return s

    # ----------------------------------------------------------------- #
    def print_stats(self) -> None:
        """Красивый вывод текущей статистики."""
        s = self.get_stats()
        print("\n" + "=" * 60)
        print("📊 СТАТИСТИКА:")
        print(f"   Успешно:     {s['generated']}")
        print(f"   Ошибок:      {s['failed']}")
        print(f"   Успешность:  {s['success_rate']:.1f}%")
        if "elapsed_seconds" in s:
            print(f"   Время:       {s['elapsed_seconds']:.0f} сек")
            print(f"   Скорость:    {s['speed_per_minute']:.1f} примеров/минуту")
        print(f"   Сбросов контекста: {s['context_resets']}")
        print(f"   Послед. ошибок: {s['consecutive_errors']}")
        print("=" * 60 + "\n")