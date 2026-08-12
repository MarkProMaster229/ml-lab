import os
import json
from safetensors.torch import load_file, save_file

source_dir = "/mnt/storage/allModel/model/models--LiquidAI--LFM2.5-2.6B-Base/snapshots/78f33a52fbe65f7665963f482179dcc3e75f0d9e/"
target_dir = "/mnt/storage/allModel/model_smart_pruned6-7-10-11-14-15"
os.makedirs(target_dir, exist_ok=True)

model_path = os.path.join(source_dir, "model.safetensors")
output_path = os.path.join(target_dir, "model.safetensors")
config_path_in = os.path.join(source_dir, "config.json")
config_path_out = os.path.join(target_dir, "config.json")

# Твоя правильная схема: аккуратное прореживание Мамбы без дыр
LAYERS_TO_REMOVE = {6, 7, 10, 11, 14, 15}

print("1. Загрузка весов Liquid LFM...")
tensors = load_file(model_path)
pruned_tensors = {}

print("2. Умное прореживание Мамбы по твоей схеме...")
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
        
        if layer_idx is None:
            pruned_tensors[key] = tensor
            continue
        
        if layer_idx in LAYERS_TO_REMOVE:
            continue
            
        # Считаем динамический сдвиг
        offset = len([x for x in LAYERS_TO_REMOVE if x < layer_idx])
        if offset > 0:
            parts[layer_part_pos] = str(layer_idx - offset)
            new_key = ".".join(parts)
            pruned_tensors[new_key] = tensor
        else:
            pruned_tensors[key] = tensor
    else:
        pruned_tensors[key] = tensor

print("3. Сохранение весов правильной сборки...")
save_file(pruned_tensors, output_path)

print("4. Обновление конфигурации...")
with open(config_path_in, "r") as f:
    config = json.load(f)

if "layer_types" in config:
    orig_layers = config["layer_types"]
    new_layers = [lt for idx, lt in enumerate(orig_layers) if idx not in LAYERS_TO_REMOVE]
    config["layer_types"] = new_layers
    config["num_hidden_layers"] = len(new_layers)

with open(config_path_out, "w") as f:
    json.dump(config, f, indent=2)

print(f"\n[УСПЕХ] Модель на {len(new_layers)} слоях собрана в: {target_dir}")
print("Мы убрали избыток, не создав дыр в геометрии!")
