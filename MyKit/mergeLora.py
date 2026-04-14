import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

lora_path = "/home/chelovek/bigWork/llama.cpp/loraForFlaffyTail4"
base_model_name = "/home/chelovek/ModelFlaffyTail_Merged_LoRAFinalyMayBe"
output_path = "/home/chelovek/ModelFlaffyTail_Merged_LoRAfiniSH"#include Merge Model

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print("Загрузка базовой модели...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)

print("Загрузка LoRA-адаптера...")
model = PeftModel.from_pretrained(base_model, lora_path)

print("Объединение весов...")
merged_model = model.merge_and_unload()

print(f"Сохранение объединённой модели в {output_path}")
merged_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
