import json
from googletrans import Translator
import time

translator = Translator()

def back_translate(text, src_lang='ru', dest_lang='de'):
    try:
        translated = translator.translate(text, src=src_lang, dest=dest_lang)

        back_translated = translator.translate(translated.text, src=dest_lang, dest=src_lang)
        return back_translated.text
    except Exception as e:
        print(f"Ошибка перевода: {e}")
        return text

def load_json_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json_data(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def augment_dataset(json_data):
    augmented_data = []
    
    for i, item in enumerate(json_data):
        print(f"Обработка записи {i+1}/{len(json_data)}")
        
        augmented_data.append(item.copy())
        
        original_text = item['text']
        augmented_text = back_translate(original_text)
        
        if augmented_text != original_text:
            augmented_item = item.copy()
            augmented_item['text'] = augmented_text
            augmented_item['augmented'] = True
            augmented_data.append(augmented_item)
    
    return augmented_data

if __name__ == "__main__":
    json_data = load_json_data('/home/chelovek/Загрузки/combined_dataset_cleaned(1).json')
    
    augmented_dataset = augment_dataset(json_data)
    
    save_json_data(augmented_dataset, 'augmented_dataset.json')
    
    print(f"Оригинальных записей: {len(json_data)}")
    print(f"Аугментированных записей: {len(augmented_dataset) - len(json_data)}")
    print(f"Всего записей: {len(augmented_dataset)}")