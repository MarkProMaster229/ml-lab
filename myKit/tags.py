#this pre-training kit!

from transformers import pipeline
import json
from transformers import pipeline
#I USING COMPLETE MODEL! THIS NOT TERRABLE!(I guess)
model_id = "blanchefort/rubert-base-cased-sentiment" 
sentiment = pipeline("sentiment-analysis", model=model_id, tokenizer=model_id)

with open("/home/chelovek/Загрузки/1234567/MydatasetT2_F.json", "r", encoding="utf-8") as f:
    data = json.load(f)

#print(data)

datas = [i['input'] for i in data]
#print(datas)
print(len(datas))

chunks = []
chunk_size = 1
for i in range(0,len(datas), chunk_size):
    chunk = datas[i:i+chunk_size]
    chunks.extend(chunk)




dataset = []
for chunk in chunks:
    #print(chunk)
    #print(sentiment(chunk))
    label = sentiment(chunk)[0]['label']
    dataset.append({
        "text": chunk,
        "label": label
    })

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)


datas = [i['target'] for i in data]
#print(datas)
print(len(datas))

chunks = []
chunk_size = 1
for i in range(0,len(datas), chunk_size):
    chunk = datas[i:i+chunk_size]
    chunks.extend(chunk)


datas = [i['target'] for i in data]
#print(datas)
print(len(datas))

dataset = []
for chunk in chunks:
    #print(chunk)
    #print(sentiment(chunk))
    label = sentiment(chunk)[0]['label']
    dataset.append({
        "text": chunk,
        "label": label
    })

with open("output2.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)


#from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
#from datasets import Dataset
#import torch
#import json
#import pandas as pd
#from tqdm import tqdm

#model_name = "sismetanin/rubert-ru-sentiment-rusentiment"

#model = AutoModelForSequenceClassification.from_pretrained(
#    model_name,
#    trust_remote_code=True,
#    dtype=torch.float16,
#    device_map="auto",
#    use_safetensors=True
#)

#tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

#sentiment = pipeline(
#    "sentiment-analysis",
#    model=model,
#    tokenizer=tokenizer,
#    truncation=True,
#    max_length=512
#)

#LABEL_MAP = {
#    "LABEL_0": "negative",
#    "LABEL_1": "neutral",
#    "LABEL_2": "positive",
#    "LABEL_3": "positive",
#    "LABEL_4": "positive"
#}

#with open("/home/chelovek/Музыка/clean_part1.json", "r", encoding="utf-8") as f:
#    data = json.load(f)

#texts = [i['text'] for i in data]
#print(f"Всего текстов: {len(texts)}")

#dataset = Dataset.from_dict({"text": texts})

#def classify_batch(batch):
#    truncated_texts = [
#        tokenizer.decode(
#            tokenizer(text, truncation=True, max_length=512)["input_ids"]
#        )
#        for text in batch["text"]
#    ]
#    results = sentiment(truncated_texts)
#    return {
#        "label": [LABEL_MAP.get(res["label"], "unknown") for res in results],
#        "score": [res["score"] for res in results]
#    }

#dataset = dataset.map(
#    classify_batch,
#    batched=True,
#    batch_size=32,
#    desc="Прогон на GPU"
#)

#df = dataset.to_pandas()
#df.to_json(
#    "/home/chelovek/Документы/work/output.json",
#    orient="records",
#    force_ascii=False,
#    indent=4
#)

#print("done")

