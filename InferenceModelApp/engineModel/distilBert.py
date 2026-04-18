from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import gc
from flask import Flask, request, jsonify, render_template_string, render_template
from flask_cors import CORS

#this file include engine only

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

class DistilBert:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
        repo_name = "MarkProMaster229/experimental_models"
        lora = "distil_Bert/loraForDistil_Bert_epoch17"

        base_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert/distilbert-base-uncased",
            num_labels=3,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        self.model = PeftModel.from_pretrained(base_model, repo_name, subfolder=lora)
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
    def predict(self, promt):
        
        self.model.eval()
        inputs = self.tokenizer(
        promt,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

        confidence = torch.softmax(logits, dim=-1).max().item()

        return self.id2label[predicted_class], confidence
    