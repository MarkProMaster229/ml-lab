import re
import json

INPUT = "/home/chelovek/Загрузки/ttelegram/two/result.json"
OUTPUT = "texts2.json"

with open(INPUT, "r", encoding="utf-8", errors="ignore") as f:
    raw = f.read()

pattern = r'"text"\s*:\s*"([^"]+)"'
texts = re.findall(pattern, raw)


cleaned = [t.replace('\n', ' ').replace('\r', ' ').strip() for t in texts if t.strip()]

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
