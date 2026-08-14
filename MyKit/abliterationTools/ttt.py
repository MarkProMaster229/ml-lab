import time
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

# ================== СИСТЕМНЫЙ ПРОМТ ==================

# =====================================================

local_model_path = "/home/chelovek/Music/456/"

print("Загрузка токенизатора...")
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    local_model_path, 
    local_files_only=True
)

print("Загрузка весов модели...")
model = AutoModelForCausalLM.from_pretrained(
    local_model_path, 
    local_files_only=True,
    torch_dtype=torch.bfloat16,  
    trust_remote_code=True,      
    device_map="auto"
)

messages = [
    {"role": "user", "content": prompt}
]

print("Применяем шаблон чата (Jinja)...")
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

print("Погнали! Генерация ответа (макс. 500 токенов)...")

# Ждем, пока GPU закончит все предварительные операции, и запускаем секундомер
if torch.cuda.is_available():
    torch.cuda.synchronize()
start_time = time.perf_counter()

outputs = model.generate(
    **inputs, 
    max_new_tokens=500,
    do_sample=True,
    temperature=0.2    
)

# Синхронизируем потоки GPU, чтобы зафиксировать точное время окончания вычислений
if torch.cuda.is_available():
    torch.cuda.synchronize()
end_time = time.perf_counter()

# Считаем количество сгенерированных токенов (хвост без учета промта)
input_len = inputs["input_ids"].shape[-1]
generated_tokens_count = len(outputs[0][input_len:])

# Расчет метрик времени
total_time = end_time - start_time
tokens_per_second = generated_tokens_count / total_time if total_time > 0 else 0

generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

print("\n[ОТВЕТ МОДЕЛИ]:")
print(generated_text)

print("\n" + "="*40)
print(f"📊 БЕНЧМАРК ГЕНЕРАЦИИ:")
print(f"⏱️ Общее время:         {total_time:.2f} сек")
print(f"🪙 Сгенерировано токенов: {generated_tokens_count}")
print(f"⚡ Скорость:            {tokens_per_second:.2f} токенов/сек")
print("="*40)