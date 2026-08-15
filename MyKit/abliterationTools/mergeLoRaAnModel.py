import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import json

# ==================== КОНФИГУРАЦИЯ ====================
BASE_MODEL_PATH = "/mnt/storage/allModel/NEW1/"
LORA_PATH = "/home/chelovek/Music/modelWork/ml-lab/mayBEthisfinaly22/checkpoint-epoch-10"
OUTPUT_PATH = "/mnt/storage/allModel/proonModel/FullModelTest"
DTYPE = "float16"  # float16, float32, bfloat16
# ======================================================

dtype_map = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16
}

print("=" * 60)
print("🔧 MERGE LORA WITH BASE MODEL")
print("=" * 60)

# Проверяем пути
if not os.path.exists(BASE_MODEL_PATH):
    raise FileNotFoundError(f"Base model not found: {BASE_MODEL_PATH}")
if not os.path.exists(LORA_PATH):
    raise FileNotFoundError(f"LoRA adapter not found: {LORA_PATH}")

print(f"\n📁 Base model: {BASE_MODEL_PATH}")
print(f"📁 LoRA adapter: {LORA_PATH}")
print(f"📁 Output path: {OUTPUT_PATH}")
print(f"📁 Dtype: {DTYPE}")

# Загружаем конфиг LoRA
print("\n📋 Loading LoRA config...")
lora_config = PeftConfig.from_pretrained(LORA_PATH)

print(f"   - LoRA r: {lora_config.r}")
print(f"   - LoRA alpha: {lora_config.lora_alpha}")
print(f"   - Target modules: {lora_config.target_modules}")
print(f"   - Task type: {lora_config.task_type}")

# Загружаем базовую модель
print("\n🔨 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    dtype=dtype_map[DTYPE],
    device_map="auto",
    trust_remote_code=True
)

print(f"   - Base model type: {type(base_model).__name__}")
print(f"   - Parameters: {base_model.num_parameters() / 1e6:.1f}M")

# Загружаем LoRA адаптер
print("\n🔌 Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

print(f"   - Trainable params (with LoRA): {model.num_parameters(only_trainable=True) / 1e6:.1f}M")

# Мержим LoRA в основную модель
print("\n⚙️ Merging LoRA into base model...")
model = model.merge_and_unload()

print("   ✓ Merge complete!")

# Сохраняем merged модель
print(f"\n💾 Saving merged model to {OUTPUT_PATH}...")
os.makedirs(OUTPUT_PATH, exist_ok=True)

model.save_pretrained(OUTPUT_PATH, safe_serialization=True)

# Сохраняем токенизатор
print("🔤 Loading and saving tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print("   ✓ Tokenizer saved")
except Exception as e:
    print(f"   ⚠️ Warning: Could not save tokenizer: {e}")

# Сохраняем информацию о мерже
merge_info = {
    "base_model": BASE_MODEL_PATH,
    "lora_adapter": LORA_PATH,
    "lora_r": lora_config.r,
    "lora_alpha": lora_config.lora_alpha,
    "target_modules": list(lora_config.target_modules),
    "dtype": DTYPE,
    "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
}

with open(os.path.join(OUTPUT_PATH, "merge_info.json"), "w", encoding="utf-8") as f:
    json.dump(merge_info, f, indent=4, ensure_ascii=False)

print(f"\n{'=' * 60}")
print("✅ MERGE COMPLETE!")
print(f"{'=' * 60}")
print(f"📁 Merged model saved to: {OUTPUT_PATH}")
print(f"💾 Check merge_info.json for details")