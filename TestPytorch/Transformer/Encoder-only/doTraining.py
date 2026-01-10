#переделать!
#вайб код за котороый мне стыдно, есть новая версия см в папке doTraing
#конечно там тоже говно код, но хотя бы не так как тут(там я как кишки вырвал мусор))) )
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import os
from transformers import AutoTokenizer


class TransformerBlock(nn.Module):
    def __init__(self, sizeVector=128, numHeads=4):
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
    def __init__(self, vocabSize=120000, maxLong=100, sizeVector=128, block=4):
        super().__init__()
        self.maxLong = maxLong 
        self.Vectorization = nn.Embedding(vocabSize, sizeVector)
        self.posEmbed = nn.Embedding(maxLong, sizeVector)
        self.layers = nn.ModuleList([
            TransformerBlock(sizeVector=sizeVector, numHeads=8)
            for _ in range(block)
        ])
        self.lmHead = nn.Linear(sizeVector, 3)  # positive, negative, neutral
    
    def forward(self, x):
        B, T = x.shape
        tok = self.Vectorization(x)
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pos = self.posEmbed(positions)
        h = tok + pos
        
        for layer in self.layers:
            h = layer(h)
        
        cls = h.mean(dim=1)
        logits = self.lmHead(cls)
        return logits

def create_simple_tokenizer(model_dir, vocab_size):
    """Простой токенизатор на основе словаря"""
    class SimpleTokenizer:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
            
        def encode(self, text, **kwargs):
            # Простая токенизация по словам
            words = text.lower().split()
            tokens = []
            for word in words:
                # Хешируем слово для получения ID
                token_id = hash(word) % (vocab_size - 1) + 1
                tokens.append(token_id)
            return tokens
        
        def __call__(self, text, **kwargs):
            tokens = self.encode(text)
            return {'input_ids': tokens}
    
    return SimpleTokenizer(vocab_size)

def finetune_with_json():
    # ПУТИ
    model_dir = '/home/chelovek/Документы/work/classifier_epoch70'
    data_path = '/home/chelovek/Загрузки/en_ru/synthetic_classification_dataset(1).json'
    
    print(f"Модель: {model_dir}")
    print(f"Данные: {data_path}")
    
    # 1. Загружаем конфигурацию модели
    config_path = os.path.join(model_dir, 'config.pth')
    config = torch.load(config_path, map_location='cpu')
    
    print("\nКонфигурация модели:")
    print(f"  vocabSize: {config['vocabSize']}")
    print(f"  maxLong: {config['maxLong']}")
    print(f"  sizeVector: {config['sizeVector']}")
    print(f"  numLayers: {config['numLayers']}")
    print(f"  num_classes: {config['num_classes']}")
    
    # 2. Загружаем веса модели
    weights_path = os.path.join(model_dir, 'model_weights.pth')
    weights = torch.load(weights_path, map_location='cpu')
    
    # 3. Создаем модель с правильными параметрами
    model = TransformerRun(
        vocabSize=config['vocabSize'],
        maxLong=config['maxLong'],
        sizeVector=config['sizeVector'],
        block=config['numLayers']  # numLayers = количество блоков
    )
    
    # 4. Загружаем веса
    model.load_state_dict(weights)
    print("✓ Модель загружена")
    
    # 5. Загружаем или создаем токенизатор
    if os.path.exists(os.path.join(model_dir, 'tokenizer.json')):
        print("Загружаю токенизатор из директории модели...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    else:
        print("Использую русский BERT токенизатор...")
        tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    
    # 6. ЗАГРУЖАЕМ ВАШИ ДАННЫЕ ИЗ JSON ФАЙЛА
    print(f"\nЗагружаю данные из: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    print(f"Загружено {len(all_data)} примеров")
    
    # Проверяем метки
    labels = [item['label'] for item in all_data]
    unique_labels = set(labels)
    print(f"Уникальные метки: {unique_labels}")
    
    # 7. Датасет для вашего формата
    class JSONDataset(Dataset):
        def __init__(self, data, tokenizer, max_length=100):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length
            # Маппинг меток из ваших данных
            self.label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
            
            # Проверяем, все ли метки известны
            for item in data:
                if item['label'] not in self.label_map:
                    print(f"⚠️ Неизвестная метка: {item['label']}")
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            text = item['text']
            label_str = item['label']
            
            # Токенизация
            if hasattr(tokenizer, '__call__'):
                # Если это callable токенизатор
                result = tokenizer(text, **{
                    'truncation': True,
                    'padding': 'max_length',
                    'max_length': self.max_length,
                    'return_tensors': 'pt'
                })
                if isinstance(result, dict) and 'input_ids' in result:
                    input_ids = result['input_ids']
                else:
                    # Простой токенизатор
                    tokens = tokenizer.encode(text)
                    tokens = tokens[:self.max_length]
                    if len(tokens) < self.max_length:
                        tokens = tokens + [0] * (self.max_length - len(tokens))
                    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
            else:
                # HuggingFace токенизатор
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids']
            
            # Метка
            label = self.label_map.get(label_str, 2)  # по умолчанию neutral
            
            return {
                'input_ids': input_ids.squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long)
            }
    
    # 8. Разделяем данные на train/val
    import random
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"\nРазделение данных:")
    print(f"  Всего: {len(all_data)}")
    print(f"  Обучающая выборка: {len(train_data)}")
    print(f"  Валидационная выборка: {len(val_data)}")
    
    # 9. Создаем датасеты и даталоадеры
    train_dataset = JSONDataset(train_data, tokenizer, max_length=config['maxLong'])
    val_dataset = JSONDataset(val_data, tokenizer, max_length=config['maxLong'])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # 10. Настройка обучения
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nУстройство: {device}")
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    # 11. Дообучение
    num_epochs = 12
    
    print("\n" + "="*50)
    print("НАЧИНАЮ ДООБУЧЕНИЕ")
    print("="*50)
    
    for epoch in range(num_epochs):
        # Обучение
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            #if batch_idx % 10 == 0:
            #    print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        train_acc = 100 * correct / total
        
        # Валидация
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        
        print(f"\nЭпоха {epoch+1}/{num_epochs}:")
        print(f"  Train - Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")
        print("-" * 50)
    
    # 12. Сохраняем дообученную модель
    output_dir = '/home/chelovek/Документы/work/classifier_finetuned'
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем веса
    torch.save(model.state_dict(), os.path.join(output_dir, 'model_weights.pth'))
    
    # Сохраняем конфигурацию
    torch.save(config, os.path.join(output_dir, 'config.pth'))
    
    # Сохраняем токенизатор если это HF токенизатор
    if hasattr(tokenizer, 'save_pretrained'):
        tokenizer.save_pretrained(output_dir)
    
    print(f"\n✓ Модель сохранена в: {output_dir}")
    
    # 13. Тестирование
    print("\n" + "="*50)
    print("ТЕСТИРОВАНИЕ")
    print("="*50)
    
    test_examples = [
        "Спасибо за отличную работу! Вы лучшие!",
        "Ужасное обслуживание, всё очень плохо",
        "Хочу уточнить некоторые детали по договору",
        "и каждый атом эфира есть шар, законченный в самом себе"
    ]
    
    model.eval()
    for text in test_examples:
        if hasattr(tokenizer, '__call__'):
            if hasattr(tokenizer, 'encode'):
                tokens = tokenizer.encode(text)
                tokens = tokens[:config['maxLong']]
                if len(tokens) < config['maxLong']:
                    tokens = tokens + [0] * (config['maxLong'] - len(tokens))
                input_ids = torch.tensor([tokens], dtype=torch.long)
            else:
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=config['maxLong'],
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids']
        else:
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=config['maxLong'],
                return_tensors='pt'
            )
            input_ids = encoding['input_ids']
        
        with torch.no_grad():
            input_ids = input_ids.to(device)
            outputs = model(input_ids)
            probs = torch.softmax(outputs, dim=1)
            
            class_names = ['positive', 'negative', 'neutral']
            predicted_idx = torch.argmax(probs, dim=1).item()
            
            print(f"\nТекст: {text[:60]}...")
            print(f"Предсказание: {class_names[predicted_idx]}")
            probs_np = probs.squeeze().cpu().numpy()
            print(f"Вероятности: positive={probs_np[0]:.3f}, negative={probs_np[1]:.3f}, neutral={probs_np[2]:.3f}")

if __name__ == "__main__":
    finetune_with_json()