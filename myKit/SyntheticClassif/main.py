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

class SyntheticDataGenerator:
    """Основной класс генератора с рандомным чередованием меток"""
    
    def __init__(self, config: Config):
        self.config = config
        self.prompt_generator = PromptGenerator(config)
        self.model_client = ModelClient(config)
        self.validator = ResponseValidator()
        self.data_manager = DataManager(config.output_filename)
        self.stats_manager = StatsManager()
        
        # Сохраняем оригинальные темы из конфига
        self.all_topics = config.topics.copy()
        self.all_scenarios = config.scenarios.copy()
        
        # Метки для классификации и их вероятности
        self.labels_with_weights = [
            ("positive", 0.4),   # 40% позитивных
            ("negative", 0.4),   # 40% негативных  
            ("neutral", 0.4)     # 40% нейтральных
        ]
        
        # Создаем список меток в соответствии с вероятностями
        self.labels_pool = []
        for label, weight in self.labels_with_weights:
            count = int(self.config.target_count * weight)
            self.labels_pool.extend([label] * count)
        
        # Перемешиваем метки для случайного порядка
        random.shuffle(self.labels_pool)
        
        print(f"📊 Создан пул из {len(self.labels_pool)} меток:")
        for label, _ in self.labels_with_weights:
            count = self.labels_pool.count(label)
            print(f"   {label}: {count} ({count/len(self.labels_pool)*100:.1f}%)")
    
    def run(self):
        """Основной цикл генерации"""
        self._print_startup_info()
        
        if not self.model_client.test_connection():
            print("❌ Не удалось подключиться к Ollama. Убедитесь что он запущен.")
            return
        
        self.stats_manager.start()
        
        try:
            self._run_random_generation()
            self._print_final_report()
            
        except KeyboardInterrupt:
            print(f"\n\n⏹️ ГЕНЕРАЦИЯ ПРЕРВАНА ПОЛЬЗОВАТЕЛЕМ")
            print(f"Данные сохранены до прерывания.")
            self.stats_manager.print_stats()
            
        except Exception as e:
            print(f"\n💥 КРИТИЧЕСКАЯ ОШИБКА: {type(e).__name__}: {e}")
            self._emergency_protocol()
    
    def _run_random_generation(self):
        """Генерация со случайным чередованием меток"""
        print(f"\n{'='*60}")
        print(f"🎰 ГЕНЕРАЦИЯ СО СЛУЧАЙНЫМ ЧЕРЕДОВАНИЕМ МЕТОК")
        print(f"{'='*60}\n")
        
        for i, label in enumerate(self.labels_pool):
            # Показываем прогресс
            if i % 10 == 0 or i == len(self.labels_pool) - 1:
                progress = (i + 1) / len(self.labels_pool) * 100
                remaining = len(self.labels_pool) - (i + 1)
                
                print(f"\n🎯 Прогресс: {i+1}/{len(self.labels_pool)} ({progress:.1f}%)")
                print(f"⏱️  Осталось: {remaining} примеров")
                print(f"📊 Успешно/Ошибок: {self.stats_manager.stats['generated']}/{self.stats_manager.stats['failed']}")
                
                # Показываем распределение уже сгенерированных меток
                if i > 0:
                    existing_data = self.data_manager.load_existing_data()
                    label_counts = {}
                    for item in existing_data:
                        lbl = item.get("label", "unknown")
                        label_counts[lbl] = label_counts.get(lbl, 0) + 1
                    
                    print(f"📈 Текущее распределение: ", end="")
                    for lbl in ["positive", "negative", "neutral"]:
                        count = label_counts.get(lbl, 0)
                        if i > 0:
                            percent = (count / (i + 1)) * 100
                            print(f"{lbl}:{count}({percent:.1f}%) ", end="")
                    print()
            
            print(f"\n🎲 Генерация #{i+1} (метка: '{label}')")
            
            self._process_single_example(label)
            
            # Задержка между запросами
            if i < len(self.labels_pool) - 1:
                time.sleep(self.config.delay_between_requests)
    
    def _process_single_example(self, required_label: str):
        """Обрабатывает один пример с требуемой меткой"""
        print("🔄 Создание промпта...")
        prompt = self.prompt_generator.generate_prompt_with_label(required_label)
        response = self.model_client.generate_response(prompt)
        
        if response:
            example = self.validator.validate(response, required_label)
            if example:
                # Форматируем в конечный формат
                formatted_example = self._format_to_target_schema(example)
                if self.data_manager.add_example(formatted_example):
                    self.stats_manager.add_success()
                else:
                    self.stats_manager.add_failure()
                    print("❌ Не удалось сохранить пример")
            else:
                self.stats_manager.add_failure()
                print("❌ Ответ не прошел валидацию")
        else:
            self.stats_manager.add_failure()
            print("❌ Ошибка получения ответа от модели")
    
    def _format_to_target_schema(self, example: Dict) -> Dict:
        """Преобразует пример в целевой формат"""
        # Случайный номер страницы
        page = random.randint(1, 100)
        
        # route_url (чаще всего sintetic)
        if random.random() > 0.7:  # 30% случаев даем реальный URL
            route_url = random.choice([
                "https://random1",
                "https://random2",
                "https://random3",
                "https://random4",
                "https://random5"
            ])
        else:
            route_url = "sintetic"
        
        return {
            "text": example.get("text", "")[:500],  # Ограничиваем длину
            "label": example.get("label", ""),
            "route_url": route_url,
            "page": page
        }
    
    def _print_startup_info(self):
        """Выводит информацию о запуске"""
        print(f"\n{'='*60}")
        print(f"🚀 ЗАПУСК ГЕНЕРАЦИИ СИНТЕТИЧЕСКИХ ДАННЫХ")
        print(f"{'='*60}")
        print(f"Цель:          {self.config.target_count} записей")
        print(f"Модель:        {self.config.model_name}")
        print(f"Файл:          {self.config.output_filename}")
        print(f"Формат:        text, label, route_url, page")
        print(f"Распределение: 40% positive, 40% negative, 20% neutral")
        print(f"Порядок:       Случайное чередование меток")
        print(f"Время старта:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        existing_count = len(self.data_manager.load_existing_data())
        print(f"📁 Найдено существующих записей: {existing_count}")
        
        # Показываем распределение меток в существующих данных
        if existing_count > 0:
            existing_data = self.data_manager.load_existing_data()
            existing_labels = {}
            for item in existing_data:
                label = item.get("label", "unknown")
                existing_labels[label] = existing_labels.get(label, 0) + 1
            
            print(f"📊 Текущее распределение меток:")
            for label in ["positive", "negative", "neutral"]:
                count = existing_labels.get(label, 0)
                percentage = (count / existing_count) * 100 if existing_count > 0 else 0
                print(f"   {label}: {count} ({percentage:.1f}%)")
    
    def _print_final_report(self):
        """Выводит финальный отчёт"""
        print(f"\n{'='*60}")
        print(f"✅ ГЕНЕРАЦИЯ ЗАВЕРШЕНА!")
        print(f"{'='*60}")
        self.stats_manager.print_stats()
        
        existing_data = self.data_manager.load_existing_data()
        total_examples = len(existing_data)
        
        # Анализ распределения меток
        label_distribution = {}
        for example in existing_data:
            label = example.get("label", "unknown")
            label_distribution[label] = label_distribution.get(label, 0) + 1
        
        print(f"\n📊 ФИНАЛЬНОЕ РАСПРЕДЕЛЕНИЕ ПО МЕТКАМ:")
        for label in ["positive", "negative", "neutral"]:
            count = label_distribution.get(label, 0)
            percentage = (count / total_examples) * 100 if total_examples > 0 else 0
            print(f"   {label}: {count} ({percentage:.1f}%)")
        
        # Показываем несколько примеров
        print(f"\n📝 ПОСЛЕДНИЕ 3 ПРИМЕРА:")
        for i, example in enumerate(existing_data[-3:]):
            text_preview = example.get("text", "")[:80] + "..." if len(example.get("text", "")) > 80 else example.get("text", "")
            print(f"   {i+1}. [{example.get('label', '?')}] {text_preview}")
        
        print(f"\n📋 ИТОГОВЫЙ ОТЧЁТ:")
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
        "target_count": 100000, 
        "delay_between_requests": 2.0,
        "output_filename": "synthetic_classification_dataset.json",
        "temperature": 0.6,
        "top_p": 0.95,
        "num_predict": 350,
        "repeat_penalty": 1.2
    }
    
    config = Config.from_dict(config_dict)
    
    # Создаём генератор
    generator = SyntheticDataGenerator(config)
    
    # Запускаем
    generator.run()


if __name__ == "__main__":
    main()