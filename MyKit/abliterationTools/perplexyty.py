import math
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from datasets import load_dataset
from tqdm import tqdm

local_model_path = "/mnt/storage/allModel/model_smart_pruned6-7-10-11-14-15"

print("Загрузка модели и токенизатора...")
tokenizer = PreTrainedTokenizerFast.from_pretrained(local_model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    local_files_only=True,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
model.eval()

# Загружаем WikiText-2 (тестовый сплит)
print("Загрузка WikiText-2 (тест)...")
test_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(test_dataset["text"])
encodings = tokenizer(text, return_tensors="pt")

# Для честного сравнения с бенчмарками используем стандартное окно 2048
max_length = 768
stride = 384

nlls = []
total_tokens = 0
prev_end_loc = 0
seq_len = encodings.input_ids.size(1)

print(f"Расчёт перплексии по {seq_len} токенам (окно={max_length}, шаг={stride})...")

with torch.no_grad():
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()

        # Маскируем уже учтённые токены, оставляем только новые
        if trg_len > 0:
            target_ids[:, :-trg_len] = -100

        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood * trg_len)
        total_tokens += trg_len
        prev_end_loc = end_loc

        if end_loc == seq_len:
            break

# Итоговый подсчёт
print("\n" + "=" * 50)
total_loss = torch.stack(nlls).sum()
avg_loss = total_loss / total_tokens
try:
    ppl = math.exp(avg_loss.item())
    print(f"📊  Объективная перплексия (PPL): {ppl:.2f}")
except OverflowError:
    ppl = float('inf')
    print("📊  Перплексия (PPL): Бесконечность (модель предсказывает крайне плохо)")

print(f"📉  Средний Loss на токен:      {avg_loss.item():.4f}")
print(f"🔢  Оценено токенов:            {total_tokens}")
print(f"🧠  Длина контекстного окна:    {max_length}")
print("=" * 50)