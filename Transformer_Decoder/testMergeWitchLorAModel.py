from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

modelPatch = "/home/chelovek/ModelFlaffyTail_Merged_LoRAfiniSH"
tokenizer = AutoTokenizer.from_pretrained(modelPatch)

model = AutoModelForCausalLM.from_pretrained(
    modelPatch,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

messages = [
    {"role": "system", "content": "Твое имя Звездочка"}
]
#python convert_hf_to_gguf.py /home/chelovek/ModelFlaffyTail_Merged_LoRAfiniSH --outfile GreateFinalyThisFlaffyTail.gguf --outtype f16
print("\n")

while True:
    test = input()
    messages.append({"role": "user", "content": test})
        
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )
        
    if hasattr(encoded, 'input_ids'):
        inputs = encoded['input_ids'].to("cuda")
    else:
        inputs = encoded.to("cuda")
        
    outputs = model.generate(
        inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id
    )
        
    response = tokenizer.decode(
        outputs[0][inputs.shape[1]:],
        skip_special_tokens=True
    )
        
    messages.append({"role": "assistant", "content": response})
        
    print(f": {response}\n")