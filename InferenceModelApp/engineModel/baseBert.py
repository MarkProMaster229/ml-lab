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

class BaseBert:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        repo_name = "MarkProMaster229/experimental_models"
        lora = "loraBERTVanila/loraForBERT7"
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "google-bert/bert-base-uncased",
            num_labels=3,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        self.model = PeftModel.from_pretrained(base_model, repo_name, subfolder=lora)
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

        id2label = {0: "negative", 1: "neutral", 2: "positive"}
        confidence = torch.softmax(logits, dim=-1).max().item()

        return id2label[predicted_class], confidence
