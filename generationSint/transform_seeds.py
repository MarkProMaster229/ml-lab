#!/usr/bin/env python3
"""
Скрипт для преобразования экспорта Telegram (JSON) в seed-диалоги
для генератора синтетического датасета (generate_dataset.py).

Фильтрует сообщения начиная с указанной даты и нарезает их
на скользящие окна заданной длины, формируя мини-диалоги.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

# -------------------- НАСТРОЙКИ --------------------
INPUT_FILE = Path("/home/chelovek/Downloads/3/ChatExport_2026-08-10/result.json")
OUTPUT_FILE = Path("seeds2.json")                # готовый список seed-диалогов
START_DATE = "2025-05-14T09:44:55"              # строгая нижняя граница
WINDOW_SIZE = 10                                  # длина мини-диалога (количество реплик)
STEP = 5                                         # шаг скользящего окна (1 = максимальное перекрытие)


def parse_date(date_str: str) -> datetime:
    """
    Парсит ISO‑дату из Telegram (может быть с 'T' или без).
    Возвращает offset‑aware datetime в UTC.
    """
    # Telegram экспорт обычно в формате "2022-08-19T21:22:28" (без часового пояса)
    # Считаем, что это локальное время, но для сравнения приводим к UTC без смещения.
    # Если есть Z или +03:00 — обработается автоматически.
    if date_str.endswith("Z"):
        date_str = date_str[:-1] + "+00:00"
    return datetime.fromisoformat(date_str).astimezone(timezone.utc)


def load_messages(file_path: Path) -> List[Dict[str, Any]]:
    """Загружает список сообщений из JSON‑файла экспорта Telegram."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ожидаем структуру с ключом "messages"
    if "messages" not in data:
        raise ValueError("В JSON отсутствует поле 'messages'")

    return data["messages"]


def filter_by_date(messages: List[Dict], start_str: str) -> List[Dict]:
    """Оставляет сообщения, у которых date >= start_str."""
    start_dt = parse_date(start_str)
    filtered = []
    for msg in messages:
        if "date" not in msg:
            continue
        try:
            msg_dt = parse_date(msg["date"])
            if msg_dt >= start_dt:
                filtered.append(msg)
        except Exception:
            # пропускаем сообщения с непарсящейся датой
            continue
    return filtered


def messages_to_lines(messages: List[Dict]) -> List[str]:
    """
    Преобразует отсортированные по времени сообщения в список строк
    формата "Имя: текст". Имя берётся из поля "from",
    текст — из поля "text" (обязательно).
    """
    lines = []
    for msg in messages:
        sender = msg.get("from", "Unknown")
        text = msg.get("text", "")
        if isinstance(text, list):  # иногда text_entities вместо plain text
            # на всякий случай склеиваем plain части
            text = "".join(
                part["text"] for part in text if isinstance(part, dict) and "text" in part
            )
        text = str(text).strip()
        if text:  # пустые сообщения пропускаем
            lines.append(f"{sender}: {text}")
    return lines


def create_windows(lines: List[str], window_size: int, step: int) -> List[List[str]]:
    """
    Нарезает последовательность реплик на скользящие окна.
    Каждое окно — список из window_size строк.
    """
    windows = []
    for start in range(0, len(lines) - window_size + 1, step):
        window = lines[start : start + window_size]
        windows.append(window)
    return windows


def main():
    print(f"Читаем {INPUT_FILE} ...")
    all_messages = load_messages(INPUT_FILE)
    print(f"Всего сообщений в файле: {len(all_messages)}")

    # Фильтрация по дате
    filtered = filter_by_date(all_messages, START_DATE)
    print(f"После фильтрации (дата >= {START_DATE}): {len(filtered)}")

    if not filtered:
        print("Нет сообщений, удовлетворяющих условию. Выход.")
        sys.exit(1)

    # Превращаем в строки "Имя: текст"
    lines = messages_to_lines(filtered)
    print(f"Валидных текстовых реплик: {len(lines)}")

    # Создаём окна-диалоги
    seed_dialogs = create_windows(lines, WINDOW_SIZE, STEP)
    print(f"Сгенерировано seed-диалогов (окон длиной {WINDOW_SIZE}): {len(seed_dialogs)}")

    # Сохраняем результат
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(seed_dialogs, f, ensure_ascii=False, indent=2)

    print(f"Готово. Результат сохранён в {OUTPUT_FILE}")
    print("Пример первого диалога:")
    for line in seed_dialogs[0]:
        print(f"  {line}")


if __name__ == "__main__":
    main()