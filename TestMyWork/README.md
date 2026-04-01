### Описание данных

Комментарии собраны из открытых источников. Разметка выполнена автоматически с использованием LLM. Проект имеет исключительно исследовательский характер, разработчик не участвовал в ручной разметке и не выражает личного мнения.
Содержимое комментариев отражает личные взгляды их авторов.

Структура данных:
```json
[
  {
    "text": "комментарий",
    "label": "negative|neutral|positive"
  }
]
```

```
Метка      Количество    Доля 
negative     234        58.5% 
neutral      129        32.2% 
positive     37         9.2%
```

### Загрузка модели
По умолчанию используется адаптер с минимальным loss на валидации например - 
```python

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
repo_name = "MarkProMaster229/experimental_models"
lora = "ForRoberta_models/loraForROBERTA_epoch8"

base_model = AutoModelForSequenceClassification.from_pretrained(
    "FacebookAI/xlm-roberta-base",
    num_labels=3,
    torch_dtype=torch.float32,
    device_map="cpu"
)

model = PeftModel.from_pretrained(base_model, repo_name, subfolder=lora)

```
### Использование других адаптеров
Для тестирования можно использовать другие эпохи адаптера из репозитория например -
```python
lora = "ForRoberta_models/loraForROBERTA_epoch4"
```
---
