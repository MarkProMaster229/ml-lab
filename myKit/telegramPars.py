import re
import json

INPUT = "/home/chelovek/Загрузки/1234567/2ch_all_posts.json"
OUTPUT = "texts2222_2ch.json"

with open(INPUT, "r", encoding="utf-8", errors="ignore") as f:
    raw = f.read()

pattern = r'"text"\s*:\s*"([^"]+)"'
texts = re.findall(pattern, raw)

cleaned = []
for t in texts:
    t = t.replace('\n', ' ').replace('\r', ' ').strip()
    if t and len(t) <= 130:
        cleaned.append(t)

seen = set()
unique = []
for t in cleaned:
    if t not in seen:
        seen.add(t)
        unique.append(t)

result = [{"text": t} for t in unique]

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("extracted:", len(result), "unique texts")
