import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import json

# ------------------------------
# Конфигурация
# ------------------------------
source_dir = "/mnt/storage/allModel/model/models--LiquidAI--LFM2.5-2.6B-Base/snapshots/78f33a52fbe65f7665963f482179dcc3e75f0d9e/"
target_dir = "/mnt/storage/allModel/model_adaptive_prunedtest"
os.makedirs(target_dir, exist_ok=True)

LAYERS_TO_PRUNE = {15, 16, 17, 18, 19, 20, 21, 22}

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 5e-6
L1_LAMBDA = 1e-4
MAX_STEPS = 500
ALPHA_THRESHOLD = 0.01

# Изменили путь на JSON-файл с диалогами
DATASET_PATH = "/home/chelovek/Downloads/dataForPr.json"
BLOCK_SIZE = 512

# ------------------------------
# Класс-обёртка
# ------------------------------
class GatedWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, *args, **kwargs):
        residual = x
        out = self.layer(x, *args, **kwargs)
        return residual + self.alpha * (out - residual)

# ------------------------------
# НОВЫЙ ДАТАСЕТ ДЛЯ JSON С INPUT/TARGET
# ------------------------------
class JSONDialogDataset(Dataset):
    """
    Читает JSON-файл с массивом объектов {"input": "...", "target": "..."}.
    Форматирует каждый диалог как:
        User: {input}
        Assistant: {target}
    и токенизирует, нарезает на блоки.
    """
    def __init__(self, file_path, tokenizer, block_size):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Собираем все диалоги в один текстовый поток
        full_text = ""
        for item in data:
            inp = item.get("input", "").strip()
            target = item.get("target", "").strip()
            if not inp and not target:
                continue
            full_text += f"User: {inp}\nAssistant: {target}\n\n"
        
        tokens = tokenizer.encode(full_text)
        self.blocks = []
        # Нарезаем на блоки, гарантируя, что последний блок полный
        for i in range(0, len(tokens) - block_size + 1, block_size):
            self.blocks.append(torch.tensor(tokens[i:i+block_size], dtype=torch.long))

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]

# ------------------------------
# Загрузка модели
# ------------------------------
print("Загружаю модель...")
model = AutoModelForCausalLM.from_pretrained(
    source_dir,
    local_files_only=True,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)

# ------------------------------
# Оборачиваем слои
# ------------------------------
gated_alphas = []
layers = model.model.layers

for idx in LAYERS_TO_PRUNE:
    wrapped = GatedWrapper(layers[idx])
    layers[idx] = wrapped
    gated_alphas.append(wrapped.alpha)

print(f"Обёрнуто слоёв: {len(gated_alphas)}")

# ------------------------------
# Токенизатор и даталоадер
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(source_dir, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Используем новый датасет
dataset = JSONDialogDataset(DATASET_PATH, tokenizer, BLOCK_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------------------------------
# Оптимизатор и заморозка
# ------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for name, param in model.named_parameters():
    param.requires_grad = ('alpha' in name)

# Собираем только alpha в список
alpha_params = [p for name, p in model.named_parameters() if 'alpha' in name]
print(f"Обучаемых alpha-параметров: {len(alpha_params)}")

optimizer = torch.optim.AdamW(alpha_params, lr=1e-2)  # был 5e-6, стал 1e-2

# ------------------------------
# Обучение
# ------------------------------
model.train()
global_step = 0

print("Начинаем плавное затухание...")
progress = tqdm(total=MAX_STEPS)

while global_step < MAX_STEPS:
    for batch in dataloader:
        input_ids = batch.to(model.device)
        outputs = model(input_ids, labels=input_ids)
        loss_ce = outputs.loss

        l1_penalty = sum(torch.abs(alpha) for alpha in gated_alphas)
        loss = loss_ce + L1_LAMBDA * l1_penalty

        loss.backward()
        # После loss.backward(), внутри цикла обучения
        if global_step == 0 or global_step % 10 == 0:
            print(f"\n--- Шаг {global_step} ---")
            for i, alpha in enumerate(gated_alphas):
                if alpha.grad is not None:
                    print(f"  alpha[{i}] grad = {alpha.grad.item():.8f}, value = {alpha.item():.4f}")
                else:
                    print(f"  alpha[{i}] grad = None! Градиент не доходит!")

        if (global_step + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            # Выводим alpha только когда реально обновились
            alpha_values = [f"{a.item():.4f}" for a in gated_alphas]
            print(f"\nStep {global_step+1}: alpha обновлены = {alpha_values}")

        progress.update(1)
        global_step += 1

        if global_step % 50 == 0:
            alpha_values = [f"{a.item():.4f}" for a in gated_alphas]
            print(f"\nStep {global_step}: alpha = {alpha_values}, loss = {loss.item():.4f}")

        if global_step >= MAX_STEPS:
            break

progress.close()

# ------------------------------
# Фиксация результатов
# ------------------------------
dead_indices = []
surviving_indices = []

for i, alpha in enumerate(gated_alphas):
    orig_idx = sorted(LAYERS_TO_PRUNE)[i]
    if alpha.item() < ALPHA_THRESHOLD:
        dead_indices.append(orig_idx)
    else:
        surviving_indices.append(orig_idx)

print(f"Слои для удаления (alpha < {ALPHA_THRESHOLD}): {dead_indices}")
print(f"Слои, которые остались живы: {surviving_indices}")

# ------------------------------
# Сохранение обрезанной модели
# ------------------------------
final_remove = set(dead_indices)
tensors = load_file(os.path.join(source_dir, "model.safetensors"))
pruned_tensors = {}

for key, tensor in tensors.items():
    if "model.layers." in key:
        parts = key.split(".")
        layer_idx = None
        layer_part_pos = None
        for i, part in enumerate(parts):
            if part.isdigit():
                layer_idx = int(part)
                layer_part_pos = i
                break
        if layer_idx is None or layer_idx in final_remove:
            continue
        offset = len([x for x in final_remove if x < layer_idx])
        parts[layer_part_pos] = str(layer_idx - offset)
        new_key = ".".join(parts)
        pruned_tensors[new_key] = tensor
    else:
        pruned_tensors[key] = tensor

output_path = os.path.join(target_dir, "model.safetensors")
save_file(pruned_tensors, output_path)

with open(os.path.join(source_dir, "config.json"), "r") as f:
    config = json.load(f)

if "layer_types" in config:
    orig_layers = config["layer_types"]
    new_layers = [lt for idx, lt in enumerate(orig_layers) if idx not in final_remove]
    config["layer_types"] = new_layers
    config["num_hidden_layers"] = len(new_layers)

with open(os.path.join(target_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"Финальная модель без {len(final_remove)} слоёв сохранена в {target_dir}")