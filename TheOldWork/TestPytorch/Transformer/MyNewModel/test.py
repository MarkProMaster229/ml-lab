#This file is entirely AI generated.
#этот файл полностью генерирован ИИ.

import torch
import argparse
from decoderOnly import TransformerRun
from transformers import AutoTokenizer

class ChatBot:
    def __init__(self, model_path="trained_model"):
        """Загружает сохраненную модель"""
        print(f"🤖 Загружаю модель из {model_path}...")
        
        # 1. Загружаем конфигурацию (безопасно)
        self.config = torch.load(f"{model_path}/config.pth", weights_only=True)
        print(f"📊 Конфигурация модели: {self.config}")
        
        # 2. Загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 3. Создаем модель с теми же параметрами
        self.model = TransformerRun(
            vocabSize=self.config['vocabSize'],
            maxLong=self.config['maxLong'],
            sizeVector=self.config['sizeVector'],
            block=self.config['numLayers']
        )
        
        # 4. Загружаем веса (безопасно)
        self.model.load_state_dict(
            torch.load(f"{model_path}/model_weights.pth", 
                      map_location='cpu', weights_only=True)
        )
        
        # 5. Настройки
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Режим оценки
        
        print("✅ Модель загружена!")
        print(f"💻 Устройство: {self.device}")
        print(f"📚 Размер словаря: {self.config['vocabSize']}")
        print(f"🔤 maxLong: {self.config['maxLong']}")
    
    def generate(self, prompt, max_length=100, temperature=0.7, top_k=50):
        """Генерирует ответ на промпт"""
        # 1. Токенизируем промпт
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=min(self.config['maxLong'] - max_length, 512)
        )
        input_ids = inputs["input_ids"].to(self.device)
        
        # 2. Генерируем токены
        generated_ids = input_ids.clone()
        
        with torch.no_grad():  # Отключаем вычисление градиентов
            for _ in range(max_length):
                # Прямой проход
                outputs = self.model(generated_ids)
                
                # Берем логиты для последнего токена
                next_token_logits = outputs[0, -1, :] / temperature
                
                # Top-k sampling (улучшает качество)
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Применяем softmax для получения вероятностей
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # Выбираем следующий токен
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Добавляем к сгенерированной последовательности
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                
                # Останавливаемся на спец-токенах
                if self.tokenizer.eos_token_id and next_token.item() == self.tokenizer.eos_token_id:
                    break
                if self.tokenizer.sep_token_id and next_token.item() == self.tokenizer.sep_token_id:
                    break
                if next_token.item() == self.tokenizer.pad_token_id:
                    break
        
        # 3. Декодируем обратно в текст
        full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Отделяем ответ от промпта
        if prompt in full_text:
            response = full_text[len(prompt):].strip()
        else:
            response = full_text.strip()
        
        return response
    
    def interactive_chat(self):
        """Интерактивный чат с моделью в бесконечном цикле"""
        print("\n" + "="*60)
        print("🤖 ЧАТ-БОТ ЗАПУЩЕН!".center(60))
        print("="*60)
        print("📝 Команды:")
        print("  /exit, /quit, /q - выйти из чата")
        print("  /clear - очистить историю")
        print("  /temp X - установить температуру (0.1-2.0)")
        print("  /len X - установить длину ответа (10-200)")
        print("  /topk X - установить top-k sampling (0-100)")
        print("="*60)
        
        # Настройки по умолчанию
        temperature = 0.7
        max_length = 100
        top_k = 50
        history = []
        
        while True:
            try:
                # Получаем промпт от пользователя
                user_input = input("\n👤 Ты: ").strip()
                
                # Обработка команд
                if user_input.lower() in ['/exit', '/quit', '/q', 'exit', 'quit', 'q']:
                    print("👋 До свидания!")
                    break
                
                elif user_input.lower() == '/clear':
                    history.clear()
                    print("🧹 История очищена!")
                    continue
                
                elif user_input.lower().startswith('/temp '):
                    try:
                        temp = float(user_input.split()[1])
                        if 0.1 <= temp <= 2.0:
                            temperature = temp
                            print(f"🌡️ Температура установлена: {temperature}")
                        else:
                            print("❌ Температура должна быть от 0.1 до 2.0")
                    except:
                        print("❌ Использование: /temp 0.7")
                    continue
                
                elif user_input.lower().startswith('/len '):
                    try:
                        length = int(user_input.split()[1])
                        if 10 <= length <= 200:
                            max_length = length
                            print(f"📏 Длина ответа установлена: {max_length}")
                        else:
                            print("❌ Длина должна быть от 10 до 200")
                    except:
                        print("❌ Использование: /len 100")
                    continue
                
                elif user_input.lower().startswith('/topk '):
                    try:
                        k = int(user_input.split()[1])
                        if 0 <= k <= 100:
                            top_k = k
                            print(f"🎯 Top-k установлен: {top_k}")
                        else:
                            print("❌ Top-k должен быть от 0 до 100")
                    except:
                        print("❌ Использование: /topk 50")
                    continue
                
                elif not user_input:
                    continue
                
                # Сохраняем в историю
                history.append(f"👤 Ты: {user_input}")
                
                # Генерируем ответ с индикацией прогресса
                print("🤖 Бот думает...", end=" ", flush=True)
                
                try:
                    response = self.generate(user_input, max_length, temperature, top_k)
                    print(f"\n🤖 Бот: {response}")
                    history.append(f"🤖 Бот: {response}")
                except Exception as e:
                    print(f"\n❌ Ошибка генерации: {e}")
                    continue
                
            except KeyboardInterrupt:
                print("\n\n👋 До свидания!")
                break
            
            except Exception as e:
                print(f"\n❌ Неизвестная ошибка: {e}")
                continue
    
    def test_chat(self, num_turns=5):
        """Тестовый чат с предопределенными промптами"""
        test_dialogue = [
            "Привет!",
            "Как тебя зовут?",
            "Что ты умеешь?",
            "Расскажи что-нибудь интересное",
            "Пока!"
        ]
        
        print("\n" + "="*60)
        print("🧪 ТЕСТОВЫЙ ДИАЛОГ".center(60))
        print("="*60)
        
        for i, prompt in enumerate(test_dialogue[:num_turns]):
            print(f"\n👤 Ты: {prompt}")
            response = self.generate(prompt, max_length=80, temperature=0.7)
            print(f"🤖 Бот: {response}")
            
            if i < num_turns - 1:
                input("\n⏎ Нажми Enter для продолжения...")
        
        print("\n" + "="*60)
        print("✅ Тест завершен!".center(60))
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Чат-бот на основе обученной модели")
    parser.add_argument("--model", type=str, default="trained_model", 
                       help="Путь к сохраненной модели")
    parser.add_argument("--chat", action="store_true", 
                       help="Включить интерактивный режим")
    parser.add_argument("--test", action="store_true",
                       help="Запустить тестовый диалог")
    parser.add_argument("--prompt", type=str, 
                       help="Один промпт для генерации")
    parser.add_argument("--temp", type=float, default=0.7,
                       help="Температура генерации (0.1-2.0)")
    parser.add_argument("--len", type=int, default=100,
                       help="Максимальная длина ответа")
    
    args = parser.parse_args()
    
    # Создаем чат-бота
    try:
        bot = ChatBot(args.model)
    except Exception as e:
        print(f"❌ Не удалось загрузить модель: {e}")
        return
    
    # Выбираем режим работы
    if args.chat:
        # Интерактивный чат
        bot.interactive_chat()
    
    elif args.test:
        # Тестовый диалог
        bot.test_chat()
    
    elif args.prompt:
        # Одиночная генерация
        print(f"\n🎯 Промпт: {args.prompt}")
        response = bot.generate(args.prompt, max_length=args.len, temperature=args.temp)
        print(f"🤖 Ответ: {response}")
    
    else:
        # Показать помощь и запустить интерактивный режим по умолчанию
        print("\n" + "="*60)
        print("💬 ЧАТ-БОТ ГОТОВ К ОБЩЕНИЮ".center(60))
        print("="*60)
        print("\nВыберите режим:")
        print("1. Интерактивный чат")
        print("2. Тестовый диалог")
        print("3. Выход")
        
        choice = input("\nВаш выбор (1-3): ").strip()
        
        if choice == "1":
            bot.interactive_chat()
        elif choice == "2":
            bot.test_chat()
        else:
            print("👋 До свидания!")

if __name__ == "__main__":
    main()