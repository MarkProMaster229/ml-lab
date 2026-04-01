## ⚠️ ВНИМАНИЕ: АГРЕССИВНЫЙ КОНТЕНТ

### Описание

Данные представляют собой комментарии, собранные из открытых источников.
Разметка выполнена автоматически с использованием LLM.
Разработчик проекта не участвовал в ручной разметке и не выражает личного мнения.

Содержимое комментариев отражает личные взгляды их авторов.
Проект создан исключительно в исследовательских целях.

Разработчик не несёт ответственности за:
- содержание исходных комментариев
- точность автоматической разметки
- любые прямые или косвенные последствия использования данных
### Структура данных

```json
[
  {
    "text": "комментарий / comment",
    "label": "negative|neutral|positive"
  }
]
```

### Загрузка модели
По умолчанию используется адаптер с минимальным loss на валидации например - 
```python
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

base_model = AutoModelForSequenceClassification.from_pretrained(
    "FacebookAI/xlm-roberta-base",
    num_labels=3
)

model = PeftModel.from_pretrained(
    base_model, 
    "MarkProMaster229/experimental_models",
    subfolder="ForRoberta_models/loraForROBERTA_epoch8"
)
```
### Использование других адаптеров
Для воспроизводимости можно использовать другие адаптеры например -
```python


model = PeftModel.from_pretrained(
    base_model,
    "MarkProMaster229/experimental_models",
    subfolder="ForRoberta_models/loraForROBERTA_epoch4"
)
```
---
