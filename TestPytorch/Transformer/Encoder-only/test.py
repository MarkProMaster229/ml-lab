import torch
import torch.nn as nn
from transformers import AutoTokenizer
import json
import time
#fully generated ai 
# 1. Восстанавливаем архитектуру модели (должна совпадать с обученной)
class TransformerBlock(nn.Module):
    def __init__(self, sizeVector=512, numHeads=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(sizeVector)
        self.attn = nn.MultiheadAttention(sizeVector, numHeads, batch_first=True)
        self.ln2 = nn.LayerNorm(sizeVector)
        self.ff = nn.Sequential(
            nn.Linear(sizeVector, sizeVector*4),
            nn.GELU(),
            nn.Linear(sizeVector*4, sizeVector)
        )
    
    def forward(self, x, key_padding_mask=None):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + attn_out
        h = self.ln2(x)
        x = x + self.ff(h)
        return x
    
class TransformerRun(nn.Module):
    def __init__(self, vocabSize=120000, maxLong=100, sizeVector=512, block=14):
        super().__init__()
        self.maxLong = maxLong 
        self.Vectorization = nn.Embedding(vocabSize, sizeVector)
        self.posEmbed = nn.Embedding(maxLong, sizeVector)
        self.layers = nn.ModuleList([
            TransformerBlock(sizeVector=sizeVector, numHeads=8)
            for _ in range(block)
        ])
        self.lmHead = nn.Linear(sizeVector, 3)
    
    def forward(self, x):
        B, T = x.shape
        tok = self.Vectorization(x)
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pos = self.posEmbed(positions)
        h = tok + pos
        cls = h.mean(dim=1)
        logits = self.lmHead(cls)
        return logits

# 2. Класс для загрузки и тестирования модели
class ModelTester:
    def __init__(self, model_path="/home/chelovek/exper/classifier_epoch5"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Загружаем конфигурацию
        config = torch.load(f"{model_path}/config.pth", map_location="cpu")
        print("Конфигурация модели:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Создаем модель с такими же параметрами
        self.model = TransformerRun(
            vocabSize=config['vocabSize'],
            maxLong=config['maxLong'],
            sizeVector=config['sizeVector'],
            block=config['numLayers']
        ).to(self.device)
        
        # Загружаем веса
        self.model.load_state_dict(torch.load(f"{model_path}/model_weights.pth", map_location=self.device))
        self.model.eval()
        print(f"Модель загружена на {self.device}")
        
        # Загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Токенизатор загружен, vocab size: {self.tokenizer.vocab_size}")
        
        # Загружаем label mapping
        try:
            self.label_map = torch.load(f"{model_path}/label_map.pth", map_location="cpu")
            print("Label mapping:", self.label_map)
        except:
            # Стандартный mapping, если файла нет
            self.label_map = {"negative": 0, "positive": 1, "neutral": 2}
            print("Используется стандартный label mapping")
        
        # Обратный mapping для вывода
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def predict(self, text, max_length=100):
        """Предсказание для одного текста"""
        # Токенизация
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        
        # Предсказание
        with torch.no_grad():
            logits = self.model(input_ids)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
        
        # Результат
        result = {
            "text": text,
            "predicted_class": predicted_class,
            "predicted_label": self.reverse_label_map.get(predicted_class, f"unknown_{predicted_class}"),
            "probabilities": {
                "negative": probabilities[0][0].item(),
                "positive": probabilities[0][1].item(),
                "neutral": probabilities[0][2].item()
            },
            "confidence": probabilities[0][predicted_class].item()
        }
        
        return result
    
    def predict_batch(self, texts, max_length=100):
        """Предсказание для списка текстов"""
        # Токенизация батча
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        
        # Предсказание
        with torch.no_grad():
            logits = self.model(input_ids)
            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()
        
        # Формируем результаты
        results = []
        for i, text in enumerate(texts):
            results.append({
                "text": text,
                "predicted_class": int(predicted_classes[i]),
                "predicted_label": self.reverse_label_map.get(predicted_classes[i], f"unknown_{predicted_classes[i]}"),
                "probabilities": {
                    "negative": probabilities[i][0].item(),
                    "positive": probabilities[i][1].item(),
                    "neutral": probabilities[i][2].item()
                },
                "confidence": probabilities[i][predicted_classes[i]].item()
            })
        
        return results
    
    def analyze_examples(self):
        """Примеры для тестирования"""
        examples = [
            # Negative
            "этот товар полное говно, никогда больше не куплю",
            "ужасное обслуживание, всё очень плохо",
            "разочарован покупкой, не советую никому",
            
            # Positive
            "отличный продукт, всем рекомендую к покупке",
            "прекрасное качество, очень доволен",
            "лучший сервис, который я встречал",
            
            # Neutral
            "товар нормальный, ничего особенного",
            "работает как ожидалось, без сюрпризов",
            "среднего качества, можно покупать если нет альтернатив",
            
            # Твои примеры из датасета
            "в кофейне starbucks брал латте с собой кофе средней температуры сахар добавил сам",
            "покупал на вайлдберриз джинсы левис размер 32 длина 32 сидят как обычно но нужно поносить",
            "был в мфц получал справку о составе семьи очередь электронная ждал 20 минут"
        ]
        
        print("\n" + "="*80)
        print("ТЕСТИРОВАНИЕ МОДЕЛИ")
        print("="*80)
        
        results = self.predict_batch(examples)
        
        for i, result in enumerate(results):
            print(f"\nПример {i+1}:")
            print(f"  Текст: {result['text'][:80]}...")
            print(f"  Предсказание: {result['predicted_label']} (класс {result['predicted_class']})")
            print(f"  Уверенность: {result['confidence']:.2%}")
            print(f"  Вероятности: neg={result['probabilities']['negative']:.2%}, "
                  f"pos={result['probabilities']['positive']:.2%}, "
                  f"neu={result['probabilities']['neutral']:.2%}")

# 3. Интерактивное тестирование
def interactive_test(model_tester):
    print("\n" + "="*80)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("="*80)
    print("Вводи текст для классификации (или 'exit' для выхода)")
    print("-"*80)
    
    while True:
        try:
            user_input = input("\nВведите текст: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'выход']:
                print("Выход из программы")
                break
            
            if not user_input:
                print("Введите текст!")
                continue
            
            result = model_tester.predict(user_input)
            
            print(f"\nРезультат:")
            print(f"  Класс: {result['predicted_label']}")
            print(f"  Уверенность: {result['confidence']:.2%}")
            print(f"  Распределение вероятностей:")
            print(f"    Negative: {result['probabilities']['negative']:.2%}")
            print(f"    Positive: {result['probabilities']['positive']:.2%}")
            print(f"    Neutral:  {result['probabilities']['neutral']:.2%}")
            
        except KeyboardInterrupt:
            for i in range(10):
                time.sleep(0.1)
                print("I DON'T WANT TO DIE")
            break
        except Exception as e:
            print(f"\nОшибка: {e}")

# 4. Запуск тестирования
if __name__ == "__main__":
    # Укажи правильный путь к сохраненной модели
    model_path = "/home/chelovek/Документы/work/classifier7772finalycut2"
    
    try:
        # Создаем тестер
        tester = ModelTester(model_path)
        
        # Тест на заранее подготовленных примерах
        tester.analyze_examples()
        
        # Интерактивный режим
        interactive_test(tester)
        
    except FileNotFoundError as e:
        print(f"Ошибка: Не найден файл по пути {model_path}")
        print("Проверь путь. Должны быть файлы:")
        print(f"  {model_path}/model_weights.pth")
        print(f"  {model_path}/config.pth")
        print(f"  {model_path}/tokenizer_config.json (и другие файлы токенизатора)")
        
        # Покажи что есть в директории
        import os
        if os.path.exists(model_path):
            print(f"\nСодержимое {model_path}:")
            for file in os.listdir(model_path):
                print(f"  - {file}")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        import traceback
        traceback.print_exc()