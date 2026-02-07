# interactive_foxy_simple.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import random

BASE_MODEL = "/home/chelovek/work/model4b"
LORA_ADAPTERS = "/home/chelovek/work/lora_project/models/lora_adapters23/checkpoint-30"

# ================== ПРОСТОЙ СИСТЕМНЫЙ ПРОМПТ ==================
SYSTEM_PROMPT = """Ты - Звездочка, дружелюбная и эмоциональная лисичка.
Ты общаешься тепло, по-дружески, с энтузиазмом.
Отвечай кратко и по делу."""

# ================== ПРОСТЫЕ НАСТРОЙКИ ==================
GENERATION_PARAMS = {
    "max_new_tokens": 150,
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.1,
    "do_sample": True,
}
# ================== ЗАГРУЗКА ==================
def load_model():
    print("🚀 Загрузка модели...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    # Проверяем токены
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Загружаем LoRA
    model = PeftModel.from_pretrained(model, LORA_ADAPTERS)
    model.eval()
    
    print("✅ Модель загружена!")
    return model, tokenizer

# ================== ПРОСТАЯ ГЕНЕРАЦИЯ ==================
def generate_response(prompt, model, tokenizer, history=None):
    """ОЧЕНЬ простая генерация без сложных шаблонов"""
    
    # Формируем простой промпт
    full_prompt = f"{SYSTEM_PROMPT}\n\nДиалог:\n"
    
    # Добавляем историю (только последние 2 обмена)
    if history:
        recent = history[-4:] if len(history) > 4 else history
        for msg in recent:
            if msg["role"] == "user":
                full_prompt += f"Человек: {msg['content']}\n"
            else:
                full_prompt += f"Звездочка: {msg['content']}\n"
    
    # Добавляем текущий запрос
    full_prompt += f"Человек: {prompt}\nЗвездочка: "
    
    # Токенизация
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    input_length = inputs['input_ids'].shape[1]
    
    # Генерация
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=GENERATION_PARAMS["max_new_tokens"],
            temperature=GENERATION_PARAMS["temperature"],
            top_p=GENERATION_PARAMS["top_p"],
            top_k=GENERATION_PARAMS["top_k"],
            repetition_penalty=GENERATION_PARAMS["repetition_penalty"],
            do_sample=GENERATION_PARAMS["do_sample"],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Декодируем
    generated_ids = outputs[0, input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Простая очистка
    response = response.split('\n')[0].strip()
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1]
    
    # Удаляем возможные префиксы
    for prefix in ["Звездочка:", "Ответ:", "Ассистент:", "Assistant:"]:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
    
    # Если ответ слишком странный, даем fallback
    if len(response) < 3 or "мышление" in response.lower() or "нужно" in response.lower():
        responses = [
            "Привет! Я Звездочка! Рада тебе! 🌟",
            "Ой, привет! Я Звездочка!",
            "Здравствуй! Я Звездочка, твой весёлый помощник!",
        ]
        response = random.choice(responses)
    
    return response

# ================== ОЧЕНЬ ПРОСТОЙ ЧАТ ==================
def simple_chat():
    model, tokenizer = load_model()
    history = []
    
    print("\n" + "=" * 50)
    print("💬 ПРОСТОЙ ЧАТ СО ЗВЕЗДОЧКОЙ")
    print("=" * 50)
    print("Напиши 'выход' чтобы выйти")
    print("Напиши 'сброс' чтобы очистить историю")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n Т: ").strip()
            
            if not user_input:
                continue
            
            # Команды
            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("👋 Пока!")
                break
            elif user_input.lower() in ['сброс', 'clear', 'reset']:
                history.clear()
                print("🗑️ История очищена!")
                continue
            
            print("model: ", end="", flush=True)
            
            # Генерация
            response = generate_response(user_input, model, tokenizer, history)
            print(response)
            
            # Сохраняем в историю (только если ответ нормальный)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
            # Ограничиваем историю
            if len(history) > 6:
                history = history[-6:]
                
        except KeyboardInterrupt:
            print("\n🛑 Прервано")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")

# ================== ЗАПУСК ==================
if __name__ == "__main__":
    simple_chat()