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

