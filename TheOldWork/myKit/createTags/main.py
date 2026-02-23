# main.py
import time
import random
from datetime import datetime
from typing import Dict, List
from config import Config
from prompt_generator import PromptGenerator
from model_client import ModelClient
from validator import ResponseValidator
from data_manager import DataManager
from stats_manager import StatsManager

class CommentClassifier:
    """Основной класс классификатора комментариев"""
    
    def __init__(self, config: Config):
        self.config = config
        self.prompt_generator = PromptGenerator(config)
        self.model_client = ModelClient(config)
        self.validator = ResponseValidator()
        self.data_manager = DataManager(config.input_filename, config.output_filename)
        self.stats_manager = StatsManager()
    
    def run(self):
        """Основной цикл классификации"""
        self._print_startup_info()
        
        if not self.model_client.test_connection():
            print("❌ Не удалось подключиться к Ollama. Убедитесь что он запущен.")
            return
        
        self.stats_manager.start()
        
        try:
            self._classify_comments()
            self._print_final_report()
            
        except KeyboardInterrupt:
            print(f"\n\n⏹️ КЛАССИФИКАЦИЯ ПРЕРВАНА ПОЛЬЗОВАТЕЛЕМ")
            print(f"Данные сохранены до прерывания.")
            self.stats_manager.print_stats()
            
        except Exception as e:
            print(f"\n💥 КРИТИЧЕСКАЯ ОШИБКА: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    def _classify_comments(self):
        """Классифицирует все комментарии"""
        print(f"\n{'='*60}")
        print(f"🎯 КЛАССИФИКАЦИЯ КОММЕНТАРИЕВ")
        print(f"{'='*60}\n")
        
        # Загружаем входные данные
        input_data = self.data_manager.load_input_data()
        if not input_data:
            print("❌ Нет данных для классификации")
            return
        
        print(f"📥 Загружено {len(input_data)} комментариев для классификации")
        
        # Загружаем уже классифицированные данные
        classified_data = self.data_manager.load_existing_classified_data()
        print(f"📊 Уже классифицировано: {len(classified_data)}")
        
        # Получаем неклассифицированные комментарии
        unclassified = self.data_manager.get_unclassified_comments(input_data, classified_data)
        print(f"📝 Осталось классифицировать: {len(unclassified)}")
        
        if len(unclassified) == 0:
            print("✅ Все комментарии уже классифицированы!")
            return
        
        # Классифицируем по одному
        for i, comment in enumerate(unclassified):
            # Показываем прогресс
            if i % 10 == 0 or i == len(unclassified) - 1:
                progress = (i + 1) / len(unclassified) * 100
                remaining = len(unclassified) - (i + 1)
                
                print(f"\n🎯 Прогресс: {i+1}/{len(unclassified)} ({progress:.1f}%)")
                print(f"⏱️  Осталось: {remaining} комментариев")
                print(f"📊 Успешно/Ошибок: {self.stats_manager.stats['generated']}/{self.stats_manager.stats['failed']}")
            
            print(f"\n🔍 Классификация #{i+1}:")
            print(f"   Текст: {comment[:80]}..." if len(comment) > 80 else f"   Текст: {comment}")
            
            self._process_single_comment(comment)
            
            # Задержка между запросами
            if i < len(unclassified) - 1:
                time.sleep(self.config.delay_between_requests)
    
    def _process_single_comment(self, comment: str):
        """Обрабатывает классификацию одного комментария"""
        prompt = self.prompt_generator.generate_classification_prompt(comment)
        response = self.model_client.generate_response(prompt)
        
        if response:
            label = self.validator.validate_classification(response)
            if label:
                if self.data_manager.add_classified_comment(comment, label):
                    self.stats_manager.add_success()
                    print(f"✅ Классифицирован как: {label}")
                else:
                    self.stats_manager.add_failure()
                    print("❌ Не удалось сохранить результат")
            else:
                self.stats_manager.add_failure()
                print("❌ Ответ не прошел валидацию")
        else:
            self.stats_manager.add_failure()
            print("❌ Ошибка получения ответа от модели")
    
    def _print_startup_info(self):
        """Выводит информацию о запуске"""
        print(f"\n{'='*60}")
        print(f"🚀 ЗАПУСК КЛАССИФИКАЦИИ КОММЕНТАРИЕВ")
        print(f"{'='*60}")
        print(f"Модель:        {self.config.model_name}")
        print(f"Входной файл:  {self.config.input_filename}")
        print(f"Выходной файл: {self.config.output_filename}")
        print(f"Формат:        text, label")
        print(f"Метки:         positive, negative, neutral")
        print(f"Время старта:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
    
    def _print_final_report(self):
        """Выводит финальный отчёт"""
        print(f"\n{'='*60}")
        print(f"✅ КЛАССИФИКАЦИЯ ЗАВЕРШЕНА!")
        print(f"{'='*60}")
        self.stats_manager.print_stats()
        
        # Анализ результатов
        classified_data = self.data_manager.load_existing_classified_data()
        total = len(classified_data)
        
        if total == 0:
            print("❌ Нет классифицированных данных")
            return
        
        # Статистика по меткам
        label_counts = {}
        for item in classified_data:
            label = item.get("label", "unknown")
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\n📊 РАСПРЕДЕЛЕНИЕ МЕТОК:")
        for label in ["positive", "negative", "neutral"]:
            count = label_counts.get(label, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"   {label}: {count} ({percentage:.1f}%)")
        
        # Примеры
        print(f"\n📝 ПОСЛЕДНИЕ 3 КЛАССИФИЦИРОВАННЫХ КОММЕНТАРИЯ:")
        for i, item in enumerate(classified_data[-3:]):
            text_preview = item.get("text", "")[:60] + "..." if len(item.get("text", "")) > 60 else item.get("text", "")
            print(f"   {i+1}. [{item.get('label', '?')}] {text_preview}")
        
        print(f"\n📋 ИТОГОВЫЙ ОТЧЁТ:")
        print(f"   Всего классифицировано: {total}")
        print(f"   Успешных классификаций: {self.stats_manager.stats['generated']}")
        print(f"   Ошибок классификации:   {self.stats_manager.stats['failed']}")
        print(f"{'='*60}")


def main():
    """Точка входа"""
    # Конфигурация
    config_dict = {
        "model_name": "devstral-2:123b-cloud",
        "ollama_url": "http://localhost:11434/api/generate",
        "delay_between_requests": 1.5,
        "input_filename": "/home/chelovek/work/PositiveComment.json",
        "output_filename": "classified_commentsPosiriveOwO.json",
        "temperature": 0.2,
        "top_p": 0.9,
        "num_predict": 50,
        "repeat_penalty": 1.1
    }
    
    config = Config.from_dict(config_dict)
    
    # Создаём классификатор
    classifier = CommentClassifier(config)
    
    # Запускаем
    classifier.run()


if __name__ == "__main__":
    main()