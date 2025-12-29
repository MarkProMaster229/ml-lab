# main.py
import time
import random
from datetime import datetime
from config import Config
from prompt_generator import PromptGenerator
from model_client import ModelClient
from validator import ResponseValidator
from data_manager import DataManager
from stats_manager import StatsManager

class SyntheticDataGenerator:
    """Основной класс генератора с тематическими циклами"""
    
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
        self.all_industries = config.industries.copy()
        self.all_professions = config.professions.copy()
        
        # Группируем темы по категориям (используем те же что уже есть)
        self.thematic_groups = self._create_thematic_groups()
    
    def _create_thematic_groups(self):
        """Создает тематические группы из существующих тем"""
        # Автоматически группируем темы по ключевым словам
        thematic_groups = {}
        
        for topic in self.all_topics:
            category = self._categorize_topic(topic)
            if category not in thematic_groups:
                thematic_groups[category] = []
            thematic_groups[category].append(topic)
        
        # Если какая-то группа получилась слишком маленькой, объединяем
        return self._balance_thematic_groups(thematic_groups)
    
    def _categorize_topic(self, topic: str) -> str:
        """Определяет категорию темы по ключевым словам"""
        topic_lower = topic.lower()
        
        if any(word in topic_lower for word in ["программир", "технолог", "IT", "компьютер", "сеть", "база данных"]):
            return "технические"
        elif any(word in topic_lower for word in ["медицин", "здоров", "врач", "лечен", "диагноз"]):
            return "медицинские"
        elif any(word in topic_lower for word in ["бизнес", "финанс", "маркетинг", "продаж", "управлен"]):
            return "деловые"
        elif any(word in topic_lower for word in ["быт", "дом", "семья", "кулинар", "ремонт", "шопинг"]):
            return "бытовые"
        elif any(word in topic_lower for word in ["образован", "обучен", "учен", "студент", "экзамен"]):
            return "образовательные"
        elif any(word in topic_lower for word in ["искусств", "творч", "дизайн", "музык", "арт"]):
            return "творческие"
        elif any(word in topic_lower for word in ["спорт", "фитнес", "йога", "тренировк"]):
            return "спортивные"
        elif any(word in topic_lower for word in ["путешеств", "туризм", "отдых", "отпуск"]):
            return "путешествия"
        elif any(word in topic_lower for word in ["наук", "исследован", "эксперимент", "лаборатор"]):
            return "научные"
        else:
            return "разные"
    
    def _balance_thematic_groups(self, groups: dict) -> list:
        """Балансирует группы по размеру"""
        balanced_groups = []
        
        # Минимальный размер группы
        min_group_size = 4
        
        # Собираем большие группы
        for category, topics in groups.items():
            if len(topics) >= min_group_size:
                balanced_groups.append({
                    "name": category,
                    "topics": topics,
                    "scenarios": [s for s in self.all_scenarios if self._topic_matches_scenario(category, s)],
                    "industries": [i for i in self.all_industries if self._topic_matches_industry(category, i)],
                    "professions": [p for p in self.all_professions if self._topic_matches_profession(category, p)]
                })
        
        # Собираем остальные темы в одну группу "разные"
        other_topics = []
        for category, topics in groups.items():
            if len(topics) < min_group_size:
                other_topics.extend(topics)
        
        if other_topics:
            balanced_groups.append({
                "name": "разные",
                "topics": other_topics,
                "scenarios": self.all_scenarios,
                "industries": self.all_industries,
                "professions": self.all_professions
            })
        
        return balanced_groups
    
    def _topic_matches_scenario(self, category: str, scenario: str) -> bool:
        """Проверяет соответствие сценария теме"""
        scenario_lower = scenario.lower()
        
        if category == "технические":
            return any(word in scenario_lower for word in ["технич", "программ", "код", "баг", "ошибк"])
        elif category == "медицинские":
            return any(word in scenario_lower for word in ["медицин", "лечен", "диагноз", "симптом"])
        elif category == "деловые":
            return any(word in scenario_lower for word in ["делов", "переговор", "продаж", "бизнес"])
        elif category == "образовательные":
            return any(word in scenario_lower for word in ["обучен", "объяснен", "экзамен", "урок"])
        else:
            return True
    
    def _topic_matches_industry(self, category: str, industry: str) -> bool:
        """Проверяет соответствие отрасли теме"""
        industry_lower = industry.lower()
        
        if category == "технические":
            return any(word in industry_lower for word in ["IT", "технолог", "телеком"])
        elif category == "медицинские":
            return any(word in industry_lower for word in ["здоров", "медицин", "фармац"])
        elif category == "деловые":
            return any(word in industry_lower for word in ["финанс", "банк", "консалт", "маркетинг"])
        elif category == "образовательные":
            return any(word in industry_lower for word in ["образован", "наука", "исследован"])
        else:
            return True
    
    def _topic_matches_profession(self, category: str, profession: str) -> bool:
        """Проверяет соответствие профессии теме"""
        profession_lower = profession.lower()
        
        if category == "технические":
            return any(word in profession_lower for word in ["программист", "инженер", "администратор", "аналитик"])
        elif category == "медицинские":
            return any(word in profession_lower for word in ["врач", "медсестра", "фельдшер", "фармацевт"])
        elif category == "деловые":
            return any(word in profession_lower for word in ["менеджер", "маркетолог", "бухгалтер", "аналитик"])
        elif category == "образовательные":
            return any(word in profession_lower for word in ["учитель", "преподаватель", "ученый", "студент"])
        else:
            return True
    
    def run(self, mode: str = "ultra_random"):
        """Основной цикл генерации с выбором режима"""
        self._print_startup_info()
        
        if not self.model_client.test_connection():
            print("❌ Не удалось подключиться к Ollama. Убедитесь что он запущен.")
            return
        
        self.stats_manager.start()
        
        try:
            if mode == "thematic_sequential":
                self._run_thematic_generation_sequential()
            elif mode == "thematic_random":
                self._run_thematic_generation_random()
            elif mode == "ultra_random":
                self._run_ultra_random_generation()
            elif mode == "standard":
                self._run_standard_generation()
            else:
                print(f"⚠️ Неизвестный режим '{mode}', использую ultra_random")
                self._run_ultra_random_generation()
            
            self._print_final_report()
            
        except KeyboardInterrupt:
            print(f"\n\n⏹️ ГЕНЕРАЦИЯ ПРЕРВАНА ПОЛЬЗОВАТЕЛЕМ")
            print(f"Данные сохранены до прерывания.")
            self.stats_manager.print_stats()
            
        except Exception as e:
            print(f"\n💥 КРИТИЧЕСКАЯ ОШИБКА: {type(e).__name__}: {e}")
            self._emergency_protocol()
    
    def _run_thematic_generation_sequential(self):
        """Генерация с тематическими циклами (группы по порядку)"""
        print(f"\n{'='*60}")
        print(f"🎨 РЕЖИМ: ТЕМАТИЧЕСКИЕ ЦИКЛЫ (ПОСЛЕДОВАТЕЛЬНО)")
        print(f"Группы идут по порядку")
        print(f"{'='*60}\n")
        
        # Рассчитываем сколько примеров генерировать в каждой теме
        examples_per_group = max(1, self.config.target_count // len(self.thematic_groups))
        print(f"Примеров на группу: {examples_per_group}")
        
        for i, group in enumerate(self.thematic_groups):
            print(f"\n{'🎨' * 30}")
            print(f"ТЕМАТИЧЕСКАЯ ГРУППА {i+1}/{len(self.thematic_groups)}: {group['name'].upper()}")
            print(f"Тем: {len(group['topics'])} | Сценариев: {len(group['scenarios'])}")
            print(f"{'🎨' * 30}\n")
            
            # Временно меняем конфиг для этой тематической группы
            original_topics = self.config.topics
            original_scenarios = self.config.scenarios
            original_industries = self.config.industries
            original_professions = self.config.professions
            
            self.config.topics = group['topics']
            self.config.scenarios = group['scenarios']
            self.config.industries = group['industries']
            self.config.professions = group['professions']
            
            # Обновляем генератор промптов с новым конфигом
            self.prompt_generator = PromptGenerator(self.config)
            
            # Генерируем примеры для этой группы
            group_start_count = self.stats_manager.stats["generated"]
            
            while (self.stats_manager.stats["generated"] < group_start_count + examples_per_group and 
                   self.stats_manager.stats["generated"] < self.config.target_count):
                
                self._process_single_example()
                self._handle_consecutive_errors()
                
                # Задержка
                remaining_total = self.config.target_count - self.stats_manager.stats["generated"]
                remaining_in_group = (group_start_count + examples_per_group) - self.stats_manager.stats["generated"]
                
                if remaining_total > 1 and remaining_in_group > 0:
                    time.sleep(self.config.delay_between_requests)
            
            # Восстанавливаем оригинальный конфиг
            self.config.topics = original_topics
            self.config.scenarios = original_scenarios
            self.config.industries = original_industries
            self.config.professions = original_professions
            
            # Восстанавливаем генератор промптов
            self.prompt_generator = PromptGenerator(self.config)
    
    def _run_thematic_generation_random(self):
        """Генерация со случайными группами"""
        print(f"\n{'='*60}")
        print(f"🎲 РЕЖИМ: СЛУЧАЙНЫЕ ГРУППЫ")
        print(f"Группы идут в случайном порядке")
        print(f"{'='*60}\n")
        
        # Перемешиваем группы
        random.shuffle(self.thematic_groups)
        
        # Рассчитываем сколько примеров генерировать в каждой теме
        examples_per_group = max(1, self.config.target_count // len(self.thematic_groups))
        print(f"Примеров на группу: {examples_per_group}")
        
        for i, group in enumerate(self.thematic_groups):
            print(f"\n{'🎲' * 30}")
            print(f"СЛУЧАЙНАЯ ГРУППА {i+1}/{len(self.thematic_groups)}: {group['name'].upper()}")
            print(f"Тем: {len(group['topics'])} | Сценариев: {len(group['scenarios'])}")
            print(f"{'🎲' * 30}\n")
            
            # Временно меняем конфиг для этой тематической группы
            original_topics = self.config.topics
            original_scenarios = self.config.scenarios
            original_industries = self.config.industries
            original_professions = self.config.professions
            
            self.config.topics = group['topics']
            self.config.scenarios = group['scenarios']
            self.config.industries = group['industries']
            self.config.professions = group['professions']
            
            # Обновляем генератор промптов с новым конфигом
            self.prompt_generator = PromptGenerator(self.config)
            
            # Генерируем примеры для этой группы
            group_start_count = self.stats_manager.stats["generated"]
            
            while (self.stats_manager.stats["generated"] < group_start_count + examples_per_group and 
                   self.stats_manager.stats["generated"] < self.config.target_count):
                
                self._process_single_example()
                self._handle_consecutive_errors()
                
                # Задержка
                remaining_total = self.config.target_count - self.stats_manager.stats["generated"]
                remaining_in_group = (group_start_count + examples_per_group) - self.stats_manager.stats["generated"]
                
                if remaining_total > 1 and remaining_in_group > 0:
                    time.sleep(self.config.delay_between_requests)
            
            # Восстанавливаем оригинальный конфиг
            self.config.topics = original_topics
            self.config.scenarios = original_scenarios
            self.config.industries = original_industries
            self.config.professions = original_professions
            
            # Восстанавливаем генератор промптов
            self.prompt_generator = PromptGenerator(self.config)
    
    def _run_ultra_random_generation(self):
        """Ультра-рандом: каждый пример из случайной группы и случайной темы"""
        print(f"\n{'='*60}")
        print(f"🎰 УЛЬТРА-РАНДОМ РЕЖИМ")
        print(f"Каждый пример - новая случайная тема")
        print(f"{'='*60}\n")
        
        # Сохраняем оригинальные настройки
        original_topics = self.config.topics
        original_scenarios = self.config.scenarios
        original_industries = self.config.industries
        original_professions = self.config.professions
        
        while self.stats_manager.stats["generated"] < self.config.target_count:
            # 1. Случайная группа
            random_group = random.choice(self.thematic_groups)
            
            # 2. Случайная тема из этой группы
            if random_group['topics']:
                random_topic = random.choice(random_group['topics'])
            else:
                random_topic = random.choice(self.all_topics)
            
            # 3. Случайный сценарий из этой группы
            if random_group['scenarios']:
                # Временно ограничиваем сценарии
                self.config.scenarios = random.sample(random_group['scenarios'], 
                                                     min(5, len(random_group['scenarios'])))
            
            # 4. Случайная отрасль и профессия
            if random_group['industries']:
                self.config.industries = random.sample(random_group['industries'], 
                                                      min(3, len(random_group['industries'])))
            if random_group['professions']:
                self.config.professions = random.sample(random_group['professions'], 
                                                       min(3, len(random_group['professions'])))
            
            # 5. Используем только одну случайную тему
            self.config.topics = [random_topic]
            self.prompt_generator = PromptGenerator(self.config)
            
            # Выводим информацию каждые 25 примеров
            remaining = self.config.target_count - self.stats_manager.stats["generated"]
            example_num = self.stats_manager.stats["generated"] + 1
            
            if example_num % 25 == 0 or remaining <= 10:
                print(f"\n🎰 Пример #{example_num}")
                print(f"   Группа: {random_group['name']}")
                print(f"   Тема: {random_topic}")
                print(f"   Осталось: {remaining}")
            
            # Генерируем ОДИН пример
            self._process_single_example()
            self._handle_consecutive_errors()
            
            # Задержка
            if remaining > 1:
                time.sleep(self.config.delay_between_requests)
            
            # Восстанавливаем ВСЕ настройки
            self.config.topics = original_topics
            self.config.scenarios = original_scenarios
            self.config.industries = original_industries
            self.config.professions = original_professions
        
        # Восстанавливаем генератор
        self.prompt_generator = PromptGenerator(self.config)
    
    def _run_standard_generation(self):
        """Стандартная генерация без тематических циклов"""
        print(f"\n{'='*60}")
        print(f"🌀 РЕЖИМ СТАНДАРТНОЙ ГЕНЕРАЦИИ")
        print(f"Все темы вперемешку")
        print(f"{'='*60}\n")
        
        while self.stats_manager.stats["generated"] < self.config.target_count:
            self._process_single_example()
            self._handle_consecutive_errors()
            
            # Задержка
            remaining = self.config.target_count - self.stats_manager.stats["generated"]
            if remaining > 1:
                time.sleep(self.config.delay_between_requests)
    
    def _print_startup_info(self):
        """Выводит информацию о запуске"""
        print(f"\n{'='*60}")
        print(f"🚀 ЗАПУСК ГЕНЕРАЦИИ СИНТЕТИЧЕСКИХ ДАННЫХ")
        print(f"{'='*60}")
        print(f"Цель:          {self.config.target_count} записей")
        print(f"Модель:        {self.config.model_name}")
        print(f"Файл:          {self.config.output_filename}")
        print(f"Темы:          {len(self.all_topics)} категорий")
        print(f"Сценарии:      {len(self.all_scenarios)} вариантов")
        print(f"Группы:        {len(self.thematic_groups)} тематических")
        print(f"Время старта:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        existing_count = len(self.data_manager.load_existing_data())
        print(f"📁 Найдено существующих записей: {existing_count}")
        
        # Показываем тематические группы
        if hasattr(self, 'thematic_groups') and self.thematic_groups:
            print(f"\n🎯 Тематические группы:")
            for i, group in enumerate(self.thematic_groups):
                print(f"   {i+1}. {group['name']}: {len(group['topics'])} тем")
    
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
                    
                    # Периодическая статистика
                    if self.stats_manager.stats["generated"] % 50 == 0:
                        self.stats_manager.print_stats()
                else:
                    self.stats_manager.add_failure()
            else:
                self.stats_manager.add_failure()
        else:
            self.stats_manager.add_failure()
    
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
        
        # Анализ тематического распределения
        if hasattr(self, 'thematic_groups') and self.thematic_groups:
            print(f"\n📊 РАСПРЕДЕЛЕНИЕ ПО ТЕМАТИЧЕСКИМ ГРУППАМ:")
            theme_distribution = {}
            
            for example in existing_data[-self.stats_manager.stats["generated"]:]:  # Только новые
                # Пытаемся определить тему по содержимому
                content = f"{example['input']} {example['target']}".lower()
                theme_found = False
                
                for group in self.thematic_groups:
                    for topic in group['topics']:
                        if topic.lower() in content:
                            theme_distribution[group['name']] = theme_distribution.get(group['name'], 0) + 1
                            theme_found = True
                            break
                    if theme_found:
                        break
                
                if not theme_found:
                    theme_distribution['не определено'] = theme_distribution.get('не определено', 0) + 1
            
            for theme, count in theme_distribution.items():
                percentage = (count / self.stats_manager.stats["generated"]) * 100
                print(f"   {theme}: {count} ({percentage:.1f}%)")
        
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
        "target_count": 5000,
        "delay_between_requests": 2.0,
        "output_filename": "synthetic_dataset.json",
        "temperature": 0.6,
        "top_p": 0.95,
        "num_predict": 350,
        "repeat_penalty": 1.2
    }
    
    config = Config.from_dict(config_dict)
    
    # Создаём генератор
    generator = SyntheticDataGenerator(config)
    
    # Выбирай режим:
    # - "thematic_sequential": группы по очереди
    # - "thematic_random": случайные группы (по 625 примеров каждая)
    # - "ultra_random": каждый пример - новая случайная тема (рекомендую!)
    # - "standard": все темы вперемешку
    
    generator.run(mode="ultra_random")


if __name__ == "__main__":
    main()