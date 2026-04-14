import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_path = "/home/chelovek/ModelFlaffyTail_Merged_LoRAFinalyMayBe"
output_path = "/home/chelovek/ModelFlaffyTail_Quantized_int4"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

model.save_pretrained(output_path, safe_serialization=True)
tokenizer.save_pretrained(output_path)

print(f"INT4 {output_path}")