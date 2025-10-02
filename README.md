# ml-lab

**Wiki для работы:** [https://github.com/MarkProMaster229/ml-lab/wiki](https://github.com/MarkProMaster229/ml-lab/wiki)

---

## Сборка

Перейди в директорию проекта и соберите программу:

```bash
cd /mnt/storage/product/ml-lab/baseModel  # свой путь

g++ -std=c++17 -g -IGeneration -IModel -ICore \
    Core/main.cpp Core/Runner.cpp Core/BatchGenerator.cpp \
    Model/Transformer.cpp Model/AttentionLogits.cpp \
    -o Core/main.out

```

Запуск

После сборки запустити:

```bash
./Core/main.out
```

## Структура проекта
```
baseModel/
│
├── Core/                  # Ядро проекта — управление пайплайном
│   ├── main.cpp           # Точка входа, запускает весь процесс
│   ├── Runner.cpp/.hpp    # Логика запуска пайплайна, контролирует последовательность действий
│   └── BatchGenerator.cpp/.hpp  # Подготовка входных тензоров: объединяет токены, маски, позиции
│
├── Generation/            # Вспомогательные инструменты для подготовки данных
│   ├── Embedding.hpp      # Генерация и обработка embedding-ов
│   ├── MaskGenerator.hpp  # Генерация масок для тензоров
│   ├── Position.hpp       # Позиционные коды для Transformer
│   ├── Tensor.hpp         # Класс тензора: хранение и базовые операции
│   ├── Tokenizer.hpp      # Токенизация входных данных
│   └── WeightGenerator.hpp # Генерация случайных весов модели (инициализация Xavier)
│
├── Model/                 # Архитектура модели
│   ├── Transformer.cpp/.hpp # Реализация Transformer
│
├── json.hpp               # Работа с JSON (чтение/запись)
├── tensor.pt              # Данные/тензоры для тестов или обучения
├── weights.pt             # Предобученные веса модели
└── test.json              # Тестовый JSON для проверки работы кода
```

