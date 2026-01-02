import json
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import time

class BackTranslationAugmenter:
    def __init__(self):
        print("Инициализация моделей для back-translation...")
        
        # Русский -> Английский
        self.model_ru_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
        self.tokenizer_ru_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
        
        # Английский -> Русский
        self.model_en_ru = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
        self.tokenizer_en_ru = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
        
        self.model_ru_en.eval()
        self.model_en_ru.eval()
        
        # Используем GPU если доступно
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.model_ru_en.to(self.device)
            self.model_en_ru.to(self.device)
            print(f"Используется GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Используется CPU")
    
    def translate_ru_en(self, text):
        """Перевод с русского на английский"""
        try:
            inputs = self.tokenizer_ru_en(text, return_tensors="pt", 
                                         padding=True, truncation=True, 
                                         max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.model_ru_en.generate(**inputs)
            
            translated = self.tokenizer_ru_en.decode(outputs[0], skip_special_tokens=True)
            return translated.strip()
        except Exception as e:
            print(f"Ошибка перевода ru->en: {e}")
            return None
    
    def translate_en_ru(self, text):
        """Перевод с английского на русский"""
        try:
            inputs = self.tokenizer_en_ru(text, return_tensors="pt", 
                                         padding=True, truncation=True,
                                         max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.model_en_ru.generate(**inputs)
            
            translated = self.tokenizer_en_ru.decode(outputs[0], skip_special_tokens=True)
            return translated.strip()
        except Exception as e:
            print(f"Ошибка перевода en->ru: {e}")
            return None
    
    def back_translate(self, text, max_retries=2):
        """Back-translation: русский -> английский -> русский"""
        for attempt in range(max_retries):
            try:
                # Шаг 1: русский -> английский
                en_text = self.translate_ru_en(text)
                if en_text is None:
                    continue
                
                
                # Шаг 2: английский -> русский
                ru_text = self.translate_en_ru(en_text)
                if ru_text is None:
                    continue
                
                # Проверяем, что получили осмысленный текст
                if len(ru_text) > 10 and ru_text != text:
                    return ru_text
                
            except Exception as e:
                print(f"Попытка {attempt + 1} не удалась: {e}")
                time.sleep(0.5)
        
        # Если все попытки не удались, возвращаем оригинальный текст
        print(f"Не удалось выполнить back-translation для текста: {text[:50]}...")
        return text

def augment_dataset(input_path, output_path, sample_size=None):
    """
    Аугментирует датасет через back-translation
    
    Args:
        input_path: путь к исходному JSON файлу
        output_path: путь для сохранения аугментированного датасета
        sample_size: количество записей для аугментации (None = все)
    """
    # Загружаем датасет
    print(f"Загрузка датасета из {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Загружено {len(data)} записей")
    
    # Инициализируем аугментатор
    augmenter = BackTranslationAugmenter()
    
    # Если указан sample_size, берем только часть данных
    if sample_size and sample_size < len(data):
        data_to_process = data[:sample_size]
        print(f"Будет обработано {sample_size} записей (сэмплирование)")
    else:
        data_to_process = data
        print(f"Будет обработано {len(data)} записей")
    
    # Создаем новый список для аугментированных данных
    augmented_data = []
    
    # Обрабатываем каждую запись
    for i, item in enumerate(tqdm(data_to_process, desc="Аугментация")):
        try:
            # Создаем копию оригинальной записи
            augmented_item = item.copy()
            
            # Получаем оригинальный текст
            original_text = item.get('text', '')
            
            if original_text:
                # Выполняем back-translation
                augmented_text = augmenter.back_translate(original_text)
                
                # Добавляем аугментированный текст в новое поле
                augmented_item['augmented_text'] = augmented_text
                
                # Сохраняем оригинальный текст тоже
                augmented_item['original_text'] = original_text
            else:
                # Если нет текста, копируем как есть
                augmented_item['augmented_text'] = ''
                augmented_item['original_text'] = ''
            
            # Добавляем в новый датасет
            augmented_data.append(augmented_item)
            
            # Периодически сохраняем промежуточные результаты
            if (i + 1) % 10 == 0:
                print(f"Обработано {i + 1}/{len(data_to_process)} записей")
                print(f"Пример {i + 1}:")
                print(f"  Оригинал: {original_text[:80]}...")
                print(f"  Аугментированный: {augmented_text[:80]}...")
                print("-" * 50)
                
        except Exception as e:
            print(f"Ошибка при обработке записи {i}: {e}")
            # Сохраняем оригинальную запись без аугментации
            item['augmented_text'] = item.get('text', '')
            item['original_text'] = item.get('text', '')
            augmented_data.append(item)
    
    # Сохраняем аугментированный датасет
    print(f"Сохранение аугментированного датасета в {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)
    
    # Статистика
    print("\n" + "="*50)
    print("АУГМЕНТАЦИЯ ЗАВЕРШЕНА!")
    print(f"Обработано записей: {len(augmented_data)}")
    print(f"Сохранено в файл: {output_path}")
    
    # Показываем несколько примеров
    print("\nПримеры аугментации:")
    for i in range(min(3, len(augmented_data))):
        item = augmented_data[i]
        print(f"\nПример {i + 1}:")
        print(f"Оригинал: {item['original_text'][:100]}...")
        print(f"Аугментированный: {item['augmented_text'][:100]}...")
        print(f"Лейбл: {item.get('label', 'N/A')}")

def create_augmented_pairs(input_path, output_path):
    """
    Создает новый датасет где каждая запись дублируется:
    одна с оригинальным текстом, одна с аугментированным
    """
    # Загружаем аугментированный датасет
    with open(input_path, 'r', encoding='utf-8') as f:
        augmented_data = json.load(f)
    
    # Создаем парный датасет
    paired_data = []
    
    for item in augmented_data:
        # Оригинальная запись (с оригинальным текстом)
        original_item = item.copy()
        original_item['text'] = item['original_text']
        original_item['is_augmented'] = False
        paired_data.append(original_item)
        
        # Аугментированная запись
        augmented_item = item.copy()
        augmented_item['text'] = item['augmented_text']
        augmented_item['is_augmented'] = True
        paired_data.append(augmented_item)
    
    # Сохраняем
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(paired_data, f, ensure_ascii=False, indent=2)
    
    print(f"Создан парный датасет: {output_path}")
    print(f"Всего записей: {len(paired_data)}")
    print(f"Оригинальных: {len(paired_data)//2}")
    print(f"Аугментированных: {len(paired_data)//2}")

if __name__ == "__main__":

    input_file = "/home/chelovek/Загрузки/combined_dataset_cleaned(1).json"
    output_file_augmented = "/home/chelovek/Загрузки/combined_dataset_augmented.json"
    output_file_paired = "/home/chelovek/Загрузки/combined_dataset_with_augmented_pairs.json"
    
    SAMPLE_SIZE = None
    CREATE_PAIRS = True 
    
    try:
        # Аугментируем датасет
        augment_dataset(
            input_path=input_file,
            output_path=output_file_augmented,
            sample_size=SAMPLE_SIZE
        )
    
    except Exception as e:
        print(f"Ошибка при выполнении: {e}")