from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import gc
from flask import Flask, request, jsonify, render_template_string, render_template
from flask_cors import CORS

#this file include engine only
class Engine:
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
    

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# BERT family
class EngineRoBert:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

        repo_name = "MarkProMaster229/experimental_models"
        lora = "ForRoberta_models/loraForROBERTA_epoch4"

        base_model = AutoModelForSequenceClassification.from_pretrained(
            "FacebookAI/xlm-roberta-base",
            num_labels=3,
            torch_dtype=torch.float32,
            device_map="cpu"
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            repo_name,
            subfolder=lora
        )

        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

    def predict(self, prompt):
        self.model.eval()

        inputs = self.tokenizer(
            prompt,
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
        pred = torch.argmax(logits, dim=-1).item()
        confidence = torch.softmax(logits, dim=-1).max().item()
        print(self.id2label[pred], confidence)

        return self.id2label[pred], confidence

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



class Manager:
    def MyCollector(self, model):
        if hasattr(model, 'model'):
            model.model.cpu()
            del model.model
        if hasattr(model, 'tokenizer'):
            del model.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        del model
        gc.collect()

        gc.collect()

    #this Load-Use-Unload
    
    def ThisController(self, promt, MyMagicObject):
        # my magic logic
        # archAutoRegr
        if MyMagicObject == 1:
            model = Engine()
            result = model.generate(promt)
            self.MyCollector(model)
            return result
        #RoBert-xlm
        elif MyMagicObject == 2:
            model = EngineRoBert()
            result = model.predict(promt)
            self.MyCollector(model)
            return result

test3 = EngineRoBert()
test3.predict("настроение ")

#this test 
app = Flask(__name__)
CORS(app)

engine1 = Engine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_data', methods=['POST'])
def process_data():
    data = request.get_json()

    print("RAW DATA:", data)

    prompt = data.get("prompt", "")

    print("PROMPT:", prompt)

    result = engine1.generate(prompt)

    return jsonify({
        "response": result
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)