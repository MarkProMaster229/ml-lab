# main.py
import time
from config import Config
from prompt_generator import PromptGenerator
from model_client import ModelClient
from validator import ResponseValidator
from data_manager import DataManager
from stats_manager import StatsManager

class SyntheticDataGenerator:
    """Основной класс генератора"""
    
    def __init__(self, config: Config):
        self.config = config
        self.prompt_generator = PromptGenerator(config)
        self.model_client = ModelClient(config)
        self.validator = ResponseValidator()
        self.data_manager = DataManager(config.output_filename)
        self.stats_manager = StatsManager()
    
    def run(self):
        """Основной цикл генерации"""
        self._print_startup_info()
        
        if not self.model_client.test_connection():
            print("❌ Не удалось подключиться к Ollama. Убедитесь что он запущен.")
            return
        
        self.stats_manager.start()
        
        try:
            while self.stats_manager.stats["generated"] < self.config.target_count:
                self._process_single_example()
                self._handle_consecutive_errors()
                
                # Задержка
                remaining = self.config.target_count - self.stats_manager.stats["generated"]
                if remaining > 1:
                    print(f"⏸️ Пауза {self.config.delay_between_requests} секунд...")
                    time.sleep(self.config.delay_between_requests)
            
            self._print_final_report()
            
        except KeyboardInterrupt:
            print(f"\n\n⏹️ ГЕНЕРАЦИЯ ПРЕРВАНА ПОЛЬЗОВАТЕЛЕМ")
            print(f"Данные сохранены до прерывания.")
            self.stats_manager.print_stats()
            
        except Exception as e:
            print(f"\n💥 КРИТИЧЕСКАЯ ОШИБКА: {type(e).__name__}: {e}")
            self._emergency_protocol()
    
    def _print_startup_info(self):
        """Выводит информацию о запуске"""
        print(f"\n{'='*60}")
        print(f"🚀 ЗАПУСК ГЕНЕРАЦИИ СИНТЕТИЧЕСКИХ ДАННЫХ")
        print(f"{'='*60}")
        print(f"Цель:          {self.config.target_count} записей")
        print(f"Модель:        {self.config.model_name}")
        print(f"Файл:          {self.config.output_filename}")
        print(f"Задержка:      {self.config.delay_between_requests} сек")
        print(f"Время старта:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        existing_count = len(self.data_manager.load_existing_data())
        print(f"📁 Найдено существующих записей: {existing_count}")
    
    def _process_single_example(self):
        """Обрабатывает один пример"""
        remaining = self.config.target_count - self.stats_manager.stats["generated"]
        print(f"\n🎯 Осталось сгенерировать: {remaining}")
        print(f"📈 Успешно/Ошибок: {self.stats_manager.stats['generated']}/{self.stats_manager.stats['failed']}")
        
        print("🔄 Генерация примера...")
        prompt = self.prompt_generator.generate_prompt()
        response = self.model_client.generate_response(prompt)
        
        if response:
            example = self.validator.validate(response)
            if example:
                if self.data_manager.add_example(example):
                    self.stats_manager.add_success()
                else:
                    self.stats_manager.add_failure()
            else:
                self.stats_manager.add_failure()
        else:
            self.stats_manager.add_failure()
        
        # Периодическая статистика
        if self.stats_manager.stats["generated"] % 10 == 0:
            self.stats_manager.print_stats()
    
    def _handle_consecutive_errors(self):
        """Обрабатывает последовательные ошибки"""
        consecutive_errors = self.stats_manager.stats["consecutive_errors"]
        
        if consecutive_errors >= 3:
            increased_delay = self.config.delay_between_requests * 3
            print(f"⚠️ Много ошибок подряд. Увеличиваю паузу до {increased_delay} секунд...")
            time.sleep(increased_delay)
        elif consecutive_errors >= 5:
            print("⚠️ Слишком много ошибок. Делаю длинную паузу 60 секунд...")
            time.sleep(60)
            self.stats_manager.stats["consecutive_errors"] = 0
    
    def _print_final_report(self):
        """Выводит финальный отчёт"""
        print(f"\n{'='*60}")
        print(f"✅ ГЕНЕРАЦИЯ ЗАВЕРШЕНА!")
        print(f"{'='*60}")
        self.stats_manager.print_stats()
        
        existing_data = self.data_manager.load_existing_data()
        total_examples = len(existing_data)
        
        print(f"📋 ИТОГОВЫЙ ОТЧЁТ:")
        print(f"   Всего в файле:   {total_examples} записей")
        print(f"   Сгенерировано:   {self.stats_manager.stats['generated']} новых")
        print(f"   Было:            {total_examples - self.stats_manager.stats['generated']} старых")
        print(f"{'='*60}")
    
    def _emergency_protocol(self):
        """Протокол экстренной ситуации"""
        print("Попытка сохранить последние данные...")
        self.stats_manager.print_stats()


def main():
    """Точка входа"""
    # Конфигурация
    config_dict = {
        "model_name": "ministral-3:latest",
        "ollama_url": "http://localhost:11434/api/generate",
        "target_count": 5000,
        "delay_between_requests": 2.0,
        "output_filename": "synthetic_dataset.json",
        "temperature": 0.8,
        "top_p": 0.95,
        "num_predict": 350,
        "repeat_penalty": 1.2
    }
    
    config = Config.from_dict(config_dict)
    
    # Создаём и запускаем генератор
    generator = SyntheticDataGenerator(config)
    generator.run()


if __name__ == "__main__":
    # Импортируем datetime для main
    from datetime import datetime
    main()