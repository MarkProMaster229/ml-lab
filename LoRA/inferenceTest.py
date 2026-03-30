from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

model_path = "/home/chelovek/bigWork/ModelArch"

tokenizer = AutoTokenizer.from_pretrained(model_path)

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    "/home/chelovek/Рабочий стол/loraForArch3/"
)

model.eval()

prompt = "<|im_start|>user\nЭй, привет, слушай, подскажи, пожалуйста, сколько будет 8 + 8\n<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

print(tokenizer.decode(output[0]))

#<|im_start|>user
#Эй, привет, слушай, подскажи, пожалуйста, сколько будет 8 + 8
#<|im_end|>
#<|im_start|>assistant
#привет! <match>8+8</match><|im_end|>