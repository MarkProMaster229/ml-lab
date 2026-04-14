from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

lora_path = "/home/chelovek/bigWork/loraForFlaffyTail4"
base_model_name = "/home/chelovek/ModelFlaffyTail_Merged"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
print(f"модель {base_model.config.hidden_size}")

model_lora = PeftModel.from_pretrained(base_model, lora_path)
model_lora.eval()

messages = [
    {"role": "system", "content": "Твое имя Звездочка,твоя задача быть полезной"},
    {"role": "user", "content": "слушай, а как твое имя"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("Сформированный промпт:")
print(repr(prompt))

inputs = tokenizer(prompt, return_tensors="pt").to(model_lora.device)

with torch.no_grad():
    output = model_lora.generate(
        **inputs,
        max_new_tokens=110,
        do_sample=False,
        temperature=0.7,
        top_p=0.9
    )


full_response = tokenizer.decode(output[0], skip_special_tokens=False)
print(full_response)


response_only = tokenizer.decode(
    output[0][inputs["input_ids"].shape[1]:], 
    skip_special_tokens=True
)
