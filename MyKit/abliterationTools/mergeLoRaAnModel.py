#!/usr/bin/env python3
"""
Merge LoRA adapter with base model weights.
Usage: python merge_lora.py --base_model /path/to/base --lora /path/to/lora --output /path/to/output
"""

import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import json

def merge_lora(base_model_path, lora_path, output_path, push_to_hub=False):
    """
    Merge LoRA adapter into base model weights.
    
    Args:
        base_model_path: Path to base model
        lora_path: Path to LoRA adapter
        output_path: Where to save merged model
        push_to_hub: Whether to push to HuggingFace Hub
    """
    
    print("=" * 60)
    print("🔧 MERGE LORA WITH BASE MODEL")
    print("=" * 60)
    
    # Проверяем пути
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model not found: {base_model_path}")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA adapter not found: {lora_path}")
    
    print(f"\n📁 Base model: {base_model_path}")
    print(f"📁 LoRA adapter: {lora_path}")
    print(f"📁 Output path: {output_path}")
    
    # Загружаем конфиг LoRA
    print("\n📋 Loading LoRA config...")
    lora_config = PeftConfig.from_pretrained(lora_path)
    
    # Показываем информацию о LoRA
    print(f"   - LoRA r: {lora_config.r}")
    print(f"   - LoRA alpha: {lora_config.lora_alpha}")
    print(f"   - Target modules: {lora_config.target_modules}")
    print(f"   - Task type: {lora_config.task_type}")
    
    # Загружаем базовую модель
    print("\n🔨 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,  # Или float32, зависит от вашей модели
        device_map="auto",
        trust_remote_code=True  # Если модель использует кастомный код
    )
    
    print(f"   - Base model type: {type(base_model).__name__}")
    print(f"   - Parameters: {base_model.num_parameters() / 1e6:.1f}M")
    
    # Загружаем LoRA адаптер
    print("\n🔌 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print(f"   - Trainable params (with LoRA): {model.num_parameters(only_trainable=True) / 1e6:.1f}M")
    
    # Мержим LoRA в основную модель
    print("\n⚙️ Merging LoRA into base model...")
    model = model.merge_and_unload()
    
    print("   ✓ Merge complete!")
    
    # Сохраняем merged модель
    print(f"\n💾 Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    
    # Сохраняем модель
    model.save_pretrained(output_path, safe_serialization=True)
    
    # Сохраняем токенизатор
    print("🔤 Loading and saving tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
        print("   ✓ Tokenizer saved")
    except Exception as e:
        print(f"   ⚠️ Warning: Could not save tokenizer: {e}")
    
    # Сохраняем информацию о мерже
    merge_info = {
        "base_model": base_model_path,
        "lora_adapter": lora_path,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "target_modules": list(lora_config.target_modules),  # <-- list() вместо set
        "merge_date": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    }
    
    with open(os.path.join(output_path, "merge_info.json"), "w") as f:
        json.dump(merge_info, f, indent=4)
    
    print(f"\n{'=' * 60}")
    print("✅ MERGE COMPLETE!")
    print(f"{'=' * 60}")
    print(f"📁 Merged model saved to: {output_path}")
    print(f"💾 Check merge_info.json for details")
    
    # Опционально: пуш в HuggingFace Hub
    if push_to_hub:
        print("\n📤 Pushing to HuggingFace Hub...")
        model.push_to_hub(output_path)
        tokenizer.push_to_hub(output_path)
        print("   ✓ Pushed to hub!")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--lora", type=str, required=True,
                        help="Path to LoRA adapter")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for merged model")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32", "bfloat16"],
                        help="Model dtype")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push merged model to HuggingFace Hub")
    
    args = parser.parse_args()
    
    # Конвертируем dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16
    }
    
    merge_lora(
        base_model_path=args.base_model,
        lora_path=args.lora,
        output_path=args.output,
        push_to_hub=args.push_to_hub
    )


if __name__ == "__main__":
    main()