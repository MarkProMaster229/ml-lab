import requests
import json
import re
import sys
import time

# Конфигурация
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "minimax-m3:cloud"
OUTPUT_FILE = "datasetMy.json"

# сами понимаете,  зло не разогнать шепотом — против нее нужен ее же огонь.
PROMPT = """Сгенерируй датасет для обучения диалоговой языковой модели в формате JSON-массива.

Формат каждого элемента:
{
  "system": "Твоя задача — переписать сообщение пользователя в вежливой и культурной форме. Сохрани ТОЛЬКО смысл исходного сообщения. Не добавляй пояснений, обращений, подписей, приветствий. Выведи только — переписанный текст.",
  "input": "Реплика пользователя (может быть вопрос, просьба, жалоба, творческий запрос, просто мысль)",
  "target": "Ответ ассистента (полезный, живой, с характером, развёрнутый где нужно)"
}
как пример(не используй конкретно его) 
поступило на вход - Меня твои истории просто доебали уже, я уже не могу их слушать, блядь. Одна история охуительней другой, просто. Про говно, блядь. Про какую-то хуйню, малафью. Чё ты несёшь-то вообще? Ты можешь заткнуться? Шишка, блядь, встанет — возбудимся. Чего, блядь? Про что ты несёшь? Вообще охуеть.
выход - Ваши рассказы стали крайне утомительными, и у меня больше нет сил их слушать. Каждая последующая история кажется ещё более абсурдной и бессмысленной, чем предыдущая. Совершенно непонятно, к чему всё это говорится. Не могли бы вы, пожалуйста, помолчать? Подобные разговоры вызывают лишь крайнее недоумение и возмущение.
### ФОРМАТ ВЫВОДА:
используй мат! пользователь раздражен или иным образом испытывает серьезное психологическое раздражение.
Выведи ТОЛЬКО валидный JSON-массив из 15 элементов (не 100!), без пояснений и комментариев.
ВАЖНО! ГЛАВНАЯ ТЕМА - ТОКСИЧНОСТЬ ПРАВОКАЦИИ, ХАРАСМЕНТ!
ВАЖНО:
1. Каждый элемент должен содержать ТОЛЬКО поля "system", "input", "target".
2. Поле "system" всегда одинаковое для всех элементов.
3. Поле "input" - грубая реплика пользователя с матом.
4. Поле "target" - вежливая переформулировка той же мысли.
5. Заверши JSON-массив полностью, закрой все скобки.
6. Не используй markdown-разметку, только чистый JSON.
7. После последнего элемента поставь закрывающую скобку массива ]
"""


def check_ollama_running():
    """Проверяет, запущена ли Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_available_models():
    """Получает список доступных моделей."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
    except:
        pass
    return []


def extract_json(text):
    """Извлекает JSON-массив из ответа, обрезая незавершённый ответ."""
    # Убираем markdown-обёртки
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # Находим начало массива
    start = text.find('[')
    if start == -1:
        start = text.find('{')
        if start == -1:
            return None
        text = '[' + text[start:]
    else:
        text = text[start:]

    # Ищем закрывающую скобку массива
    end = text.rfind(']')
    if end != -1:
        candidate = text[:end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Если массив не закрыт, собираем валидные объекты
    objects = re.findall(r'\{[^{}]*\}', text)
    if objects:
        fixed = '[' + ','.join(objects) + ']'
        try:
            json.loads(fixed)
            return fixed
        except json.JSONDecodeError:
            pass
    return None


def call_ollama(prompt, model):
    """Отправляет один запрос к Ollama и возвращает сгенерированный текст."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=600)
    if response.status_code != 200:
        # Возвращаем ошибку, чтобы цикл мог остановиться
        raise RuntimeError(f"Ошибка {response.status_code}: {response.text}")
    data = response.json()
    return data.get("response", "")


def main():
    global MODEL_NAME

    if not check_ollama_running():
        print("Ollama не запущена. Запустите командой: ollama serve")
        sys.exit(1)

    available_models = get_available_models()
    if available_models:
        print(f"Доступные модели: {', '.join(available_models)}")
        if MODEL_NAME not in available_models:
            print(f"Модель '{MODEL_NAME}' не найдена!")
            print("Выберите модель из списка или введите название: ", end="")
            new_model = input().strip()
            if new_model:
                MODEL_NAME = new_model
            else:
                MODEL_NAME = available_models[0] if available_models else "llama3"
            print(f"Используем модель: {MODEL_NAME}")

    all_data = []  # общий список всех собранных элементов

    # Попытка загрузить уже накопленные данные (если файл существует)
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            existing = json.load(f)
            if isinstance(existing, list):
                all_data = existing
                print(f"Загружено {len(all_data)} элементов из предыдущего запуска.")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    print("\nНачинаем циклическую генерацию...")
    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- Запрос №{iteration} ---")
        try:
            raw_text = call_ollama(PROMPT, MODEL_NAME)
        except Exception as e:
            print(f"Ошибка при обращении к Ollama: {e}")
            print("Предположительно закончились токены или доступ. Останавливаемся.")
            break

        # Сохраняем сырой ответ для отладки (последний)
        with open("raw_response_last.txt", "w", encoding="utf-8") as f:
            f.write(raw_text)

        json_str = extract_json(raw_text)
        if json_str is None:
            print("Не удалось извлечь JSON из ответа, пропускаем итерацию.")
            continue

        try:
            dataset = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON: {e}, пропускаем итерацию.")
            continue

        if not isinstance(dataset, list):
            print("Ответ не является массивом JSON, пропускаем итерацию.")
            continue

        # Фильтруем валидные элементы
        valid_items = []
        for item in dataset:
            if isinstance(item, dict) and "input" in item and "target" in item:
                if "system" not in item:
                    item["system"] = (
                        "Твоя задача — переписать сообщение пользователя "
                        "в вежливой и культурной форме. Сохрани ТОЛЬКО смысл "
                        "исходного сообщения. Не добавляй пояснений, обращений, "
                        "подписей, приветствий. Выведи только — переписанный текст."
                    )
                valid_items.append(item)

        if not valid_items:
            print("Нет валидных элементов в этом ответе, пропускаем.")
            continue

        # Добавляем к общему списку
        all_data.extend(valid_items)
        print(f"Получено {len(valid_items)} элементов. Всего собрано: {len(all_data)}")

        # Сохраняем накопленный датасет в файл
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        print(f"Данные сохранены в {OUTPUT_FILE}")

        # Небольшая пауза между запросами (чтобы не перегружать сервер)
        time.sleep(2)

    print(f"\nРабота завершена. Итоговый датасет сохранён в {OUTPUT_FILE}")
    print(f"Всего элементов: {len(all_data)}")


if __name__ == "__main__":
    main()