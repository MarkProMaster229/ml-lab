from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
repo_name = "MarkProMaster229/experimental_models"
lora = "loraForArchkit/loraForArch4"

tokenizer = AutoTokenizer.from_pretrained("katanemo/Arch-Router-1.5B")

base_model = AutoModelForCausalLM.from_pretrained(
    "katanemo/Arch-Router-1.5B",
    device_map="cpu",
    torch_dtype=torch.float32
)

modelLora = PeftModel.from_pretrained(
    base_model,
    repo_name,
    subfolder=lora
)

modelLora.eval()
#если вы тестируете модель, пожалуйста сохраните диалоговую конструкцию
#запрос необходимо обрамлять специльным тегом, как на примере ниже
prompt = "<|im_start|>user\nПривет, подскажи пожалуйста - сколько будет 8 + 8\n<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to(modelLora.device)

with torch.no_grad():
    output = modelLora.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

print(tokenizer.decode(output[0]))
#standart(argmax)
#<|im_start|>user\Привет, подскажи пожалуйста - сколько будет 8 + 8
#<|im_end|>
#<|im_start|>assistant
#8 + 8 равно 16.<|im_end|>




#given thet sampling = False  - argmax
#<|im_start|>user\Привет, подскажи пожалуйста - сколько будет 8 + 8
#<|im_end|>
#<|im_start|>assistant
#привет! <match>8+8</match><|im_end|>
