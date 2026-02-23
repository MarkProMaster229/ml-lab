import json
import random
from docx import Document

input_min_len = 10
input_max_len = 30
target_max_len = 30

input_file = "/home/chelovek2/Загрузки/voina-i-mir.docx"
output_file = "MydatasetT2.json"

doc = Document(input_file)
texts = [p.text.strip().replace("\n", " ") for p in doc.paragraphs if p.text.strip()]

dataset = []

for text in texts:
    if len(text) < 50:
        continue
        
    words = text.split()
    if len(words) < input_min_len + target_max_len:
        continue
    
    max_start = len(words) - input_min_len - target_max_len
    if max_start <= 0:
        continue
        
    start_pos = random.randint(0, max_start)
    input_len = random.randint(input_min_len, min(input_max_len, len(words) - start_pos - target_max_len))
    
    input_words = words[start_pos: start_pos + input_len]
    target_words = words[start_pos + input_len: start_pos + input_len + target_max_len]
    
    if len(input_words) < 5 or len(target_words) < 5:
        continue
        
    dataset.append({
        "input": " ".join(input_words),
        "target": " ".join(target_words)
    })

random.shuffle(dataset)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"Сгенерировано {len(dataset)} пар input/target в {output_file}")