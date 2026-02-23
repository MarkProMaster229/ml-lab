# modules/main.py
import random
import time
from datetime import datetime
from config import Config
from data_manager import DataManager
from model_client import ModelClient
from prompt_generator import PromptGenerator
from validator import ResponseValidator

class SimpleGenerator:
    """Упрощенный генератор диалогов."""
    
    def __init__(self, config: Config):
        self.config = config
        self.prompt_gen = PromptGenerator(config)
        self.model = ModelClient(config)
        self.validator = ResponseValidator()
        self.data_manager = DataManager(config.output_filename)
        self.generated = 0
    
    def run(self):
        """Запускает генерацию."""
        print("=" * 60)
        print("🚀 ЗАПУСК УПРОЩЕННОЙ ГЕНЕРАЦИИ")
        print("=" * 60)
        
        # Проверка соединения
        if not self.model.test_connection():
            print("❌ Не удалось подключиться к Ollama")
            return
        
        start_time = time.time()
        
        try:
            while self.generated < self.config.target_count:
                self._generate_batch()
                time.sleep(self.config.delay_between_requests)
                
                # Статистика каждые 100 записей
                if self.generated % 100 == 0:
                    elapsed = time.time() - start_time
                    speed = self.generated / (elapsed / 60) if elapsed > 0 else 0
                    print(f"\n📊 Статистика: {self.generated}/{self.config.target_count}")
                    print(f"   Скорость: {speed:.1f} пар/мин")
        
        except KeyboardInterrupt:
            print("\n⏹️ Остановлено пользователем")
        
        print(f"\n✅ Завершено! Сгенерировано: {self.generated} пар")
    
    def _generate_batch(self):
        """Генерирует один пакет из 30 пар."""
        prompt = self.prompt_gen.generate_prompt()
        response = self.model.generate_response(prompt)
        
        if not response:
            print("⚠️ Пустой ответ от модели")
            return
        
        # Парсим JSON
        examples = self.validator.validate_batch(response)
        
        if not examples:
            print("❌ Не удалось распарсить ответ")
            return
        
        # Сохраняем
        saved = 0
        for ex in examples:
            if self.data_manager.add_example(ex):
                saved += 1
        
        self.generated += saved
        print(f"✅ Сохранено {saved} из {len(examples)}")

def main():
    """Точка входа."""
    config = Config(
        target_count=500000,
        output_filename="synthetic_dataset.json",
        temperature=0.8
    )
    
    generator = SimpleGenerator(config)
    generator.run()

if __name__ == "__main__":
    main()