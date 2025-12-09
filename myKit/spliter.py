import json
import random
from docx import Document

input_min_len = 10
input_max_len = 30
target_max_len = 30

input_file = "/home/chelovek/Загрузки/Detstvo2.docx"
output_file = "MydatasetT2.json"

doc = Document(input_file)
texts = [p.text.strip().replace("\n", " ") for p in doc.paragraphs if p.text.strip()]
text = " ".join(texts)
words = text.split()

dataset = []

pos = 0
num_words = len(words)

while pos < num_words:
    input_len_words = random.randint(input_min_len, input_max_len)
    target_len_words = target_max_len

    input_words = words[pos: pos + input_len_words]
    input_text = " ".join(input_words)

    target_start = pos + input_len_words
    target_words = words[target_start: target_start + target_len_words]
    target_text = " ".join(target_words)

    if not input_text.strip() or not target_text.strip():
        break

    dataset.append({
        "input": input_text.strip(),
        "target": target_text.strip()
    })

    pos += input_len_words

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"Сгенерировано {len(dataset)} пар input/target в {output_file}")
