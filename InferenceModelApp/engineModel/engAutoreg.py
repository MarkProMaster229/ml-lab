from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import gc
from flask import Flask, request, jsonify, render_template_string, render_template
from flask_cors import CORS

#this file include engine only
class EngineAutoreg:
    def __init__(self):

        repo_name = "MarkProMaster229/experimental_models"
        lora = "loraForArchkit/loraForArch4"

        self.tokenizer = AutoTokenizer.from_pretrained("katanemo/Arch-Router-1.5B")

        base_model = AutoModelForCausalLM.from_pretrained(
            "katanemo/Arch-Router-1.5B",
            device_map="cpu",
            torch_dtype=torch.float32
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            repo_name,
            subfolder=lora
        )

        self.model.eval()
        print("Model loaded")

    def generate(self, prompt):
        prompt = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
