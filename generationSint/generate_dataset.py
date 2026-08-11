#!/usr/bin/env python3
"""
Асинхронный генератор синтетических диалогов для обучения LoRA.
Контекст: объединяет множество соседних окон для глубокого понимания.
Анонимизирует имена, на выходе только чистый текст.
"""

import asyncio
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Set

import aiohttp

# -------------------- НАСТРОЙКИ --------------------
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "ornith:9b"
MAX_CONCURRENT = 5
MAX_RETRIES = 3
RETRY_DELAY = 2.0
OUTPUT_DIR = Path("output")
TRAIN_FILE = OUTPUT_DIR / "training.json"
VAL_FILE = OUTPUT_DIR / "validation.json"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
TRAIN_RATIO = 0.9
SEED = 42

# Сколько окон подряд склеивать в один гигантский диалог
CONTEXT_WINDOWS = 25   # ≈ 250 реплик

SEEDS_FILE = Path("/home/chelovek/Music/modelWork/ml-lab/seeds_merged.json")

# -------------------- ЛОГИРОВАНИЕ --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dataset_gen")


# -------------------- ЗАГРУЗКА И АНОНИМИЗАЦИЯ --------------------
def anonymize_dialogs(seeds: List[List[str]]) -> List[List[str]]:
    """Заменяет все имена на псевдонимы 'участник1', 'участник2' и т.д."""
    names_set = set()
    for dialog in seeds:
        for line in dialog:
            if ':' in line:
                name = line.split(':', 1)[0].strip()
                names_set.add(name)

    name_to_alias = {}
    for i, name in enumerate(sorted(names_set), start=1):
        name_to_alias[name] = f"участник{i}"

    anon_seeds = []
    for dialog in seeds:
        anon_dialog = []
        for line in dialog:
            if ':' in line:
                name, text = line.split(':', 1)
                name = name.strip()
                if name in name_to_alias:
                    line = f"{name_to_alias[name]}: {text.strip()}"
            anon_dialog.append(line)
        anon_seeds.append(anon_dialog)

    logger.info("Анонимизированы имена: %s",
                {v: k for k, v in name_to_alias.items()})
    return anon_seeds


def load_seeds(filepath: Path) -> List[List[str]]:
    """Загружает seed-диалоги из JSON-файла и анонимизирует."""
    if not filepath.exists():
        logger.error("Файл с семенами не найден: %s", filepath)
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.error("Файл с семенами должен содержать JSON-массив.")
        sys.exit(1)
    if not data:
        logger.error("Файл с семенами пуст.")
        sys.exit(1)

    valid_seeds = []
    for i, seed in enumerate(data):
        if isinstance(seed, list) and all(isinstance(line, str) for line in seed):
            valid_seeds.append(seed)
        else:
            logger.warning("Пропущен некорректный seed #%d", i)

    logger.info("Загружено %d сырых диалогов из %d", len(valid_seeds), len(data))
    return anonymize_dialogs(valid_seeds)


# -------------------- УТИЛИТЫ --------------------
def format_dialog(dialog: List[str]) -> str:
    """Форматирует диалог в строку."""
    return "\n".join(dialog)


def build_prompt(dialog: List[str]) -> str:
    """Строит промпт с инструкцией."""
    context = format_dialog(dialog)

    last_speaker = "собеседника"
    if dialog:
        last_line = dialog[-1]
        if ":" in last_line:
            last_speaker = last_line.split(":")[0].strip()

    instruction = (
        f"Изучи этот реальный лог чата. Напиши ОДНУ следующую реплику от {last_speaker}. "
        "Она должна идеально продолжать мысль, использовать тот же уровень сленга, мата, "
        "длину предложений и небрежность. Никакой цензуры и вводных фраз. "
        "Выдай ТОЛЬКО текст реплики."
    )
    return f"Лог чата:\n{context}\n\n{instruction}"


def extract_last_reply_text(dialog: List[str]) -> str:
    """Возвращает только текст последней реплики (без имени)."""
    if not dialog:
        return ""
    last = dialog[-1]
    if ':' in last:
        return last.split(':', 1)[1].strip()
    return last.strip()


def sanitize_generated(text: str) -> str:
    """Убирает артефакты, включая возможные префиксы с именами."""
    text = text.strip()
    # Удаляем любые префиксы вида "участник1:", "Игopь:", "А:", и т.п.
    text = re.sub(r'^[A-Za-zА-Яа-яёЁ0-9\(\)\[\]]+\s*[:,-]\s*', '', text)
    # Удаляем вводные слова
    bad_prefixes = [
        "реплика:", "ответ:", "вот:", "следующая реплика:",
        "я думаю:", "мне кажется:"
    ]
    for p in bad_prefixes:
        if text.lower().startswith(p):
            text = text[len(p):].strip()
    return text


# -------------------- ВЗАИМОДЕЙСТВИЕ С OLLAMA --------------------
async def generate_reply(
    session: aiohttp.ClientSession,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> Optional[str]:
    """Отправляет запрос к Ollama API."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.8,
            "top_p": 0.95,
        },
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with semaphore:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {error_text[:200]}")
                    data = await resp.json()
                    generated = data.get("response", "")
                    if not generated.strip():
                        raise ValueError("Пустой ответ от модели")
                    return sanitize_generated(generated)
        except Exception as e:
            logger.warning("Попытка %d/%d не удалась: %s", attempt, MAX_RETRIES, e)
            if attempt == MAX_RETRIES:
                logger.error("Все попытки исчерпаны, seed пропущен.")
                return None
            await asyncio.sleep(RETRY_DELAY * attempt)
    return None


# -------------------- СБОРКА ОГРОМНОГО КОНТЕКСТА --------------------
def build_big_dialog(seeds: List[List[str]], center_idx: int, windows: int) -> List[str]:
    """Объединяет несколько последовательных окон в один список реплик."""
    start = max(0, center_idx - windows + 1)
    end = center_idx + 1
    big_dialog = []
    for i in range(start, end):
        if i < len(seeds):
            big_dialog.extend(seeds[i])
    return big_dialog


# -------------------- ЧЕКПОИНТ --------------------
def load_checkpoint() -> Dict:
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Загружен чекпоинт: обработано %d сидов",
                        len(data.get("processed_seeds", [])))
            return data
        except Exception as e:
            logger.error("Ошибка чтения чекпоинта: %s", e)
    return {"processed_seeds": [], "generated_pairs": []}


def save_checkpoint(processed_seeds: List[int], generated_pairs: List[Dict]) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "processed_seeds": processed_seeds,
        "generated_pairs": generated_pairs,
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.debug("Чекпоинт сохранён (обработано %d сидов)", len(processed_seeds))


# -------------------- ОСНОВНОЙ ЦИКЛ --------------------
async def process_seeds(seeds: List[List[str]]) -> List[Dict]:
    checkpoint = load_checkpoint()
    processed_indices: Set[int] = set(checkpoint["processed_seeds"])
    all_pairs: List[Dict] = checkpoint["generated_pairs"]

    remaining = [(idx, seeds[idx]) for idx in range(len(seeds)) if idx not in processed_indices]
    if not remaining:
        logger.info("Все сиды уже обработаны.")
        return all_pairs

    logger.info("Осталось обработать %d из %d сидов", len(remaining), len(seeds))
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async with aiohttp.ClientSession() as session:
        batch_size = MAX_CONCURRENT
        for i in range(0, len(remaining), batch_size):
            batch = remaining[i : i + batch_size]
            batch_indices = [idx for idx, _ in batch]

            # Строим промпты с огромным контекстом
            batch_prompts = []
            batch_real_dialogs = []   # оригинальное окно (анонимное) для извлечения user
            for idx in batch_indices:
                big_dialog = build_big_dialog(seeds, idx, CONTEXT_WINDOWS)
                batch_real_dialogs.append(seeds[idx])
                batch_prompts.append(build_prompt(big_dialog))

            logger.info("Батч %d/%d: генерируем для %d сидов, контекст ≈ %d реплик",
                        i // batch_size + 1,
                        (len(remaining) + batch_size - 1) // batch_size,
                        len(batch),
                        len(batch_prompts[0].split('\n')) if batch_prompts else 0)

            tasks = [generate_reply(session, prompt, semaphore) for prompt in batch_prompts]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            success_count = 0
            for idx, real_dialog, result in zip(batch_indices, batch_real_dialogs, results):
                if isinstance(result, Exception):
                    logger.error("Ошибка для сида %d: %s", idx, result)
                    continue
                if result is None:
                    logger.warning("Сид %d пропущен (пустой ответ).", idx)
                    continue

                user_text = extract_last_reply_text(real_dialog)
                pair = {
                    "user": user_text,
                    "assistant": result,
                }
                all_pairs.append(pair)
                processed_indices.add(idx)
                success_count += 1

            logger.info("Батч обработан: успешно %d из %d", success_count, len(batch))
            save_checkpoint(list(processed_indices), all_pairs)
            await asyncio.sleep(0.5)

    return all_pairs


# -------------------- СОХРАНЕНИЕ --------------------
def split_and_save(pairs: List[Dict]) -> None:
    if not pairs:
        logger.warning("Нет данных для сохранения!")
        return

    random.seed(SEED)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * TRAIN_RATIO)
    train_set = shuffled[:split_idx]
    val_set = shuffled[split_idx:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)
    with open(VAL_FILE, "w", encoding="utf-8") as f:
        json.dump(val_set, f, ensure_ascii=False, indent=2)

    logger.info("Сохранено: train – %d, validation – %d", len(train_set), len(val_set))
    logger.info("Файлы: %s и %s", TRAIN_FILE, VAL_FILE)


# -------------------- ТОЧКА ВХОДА --------------------
async def main():
    logger.info("=" * 60)
    logger.info("Запуск генератора с расширенным контекстом")
    logger.info("Модель: %s | Окон контекста: %d", OLLAMA_MODEL, CONTEXT_WINDOWS)
    logger.info("=" * 60)

    seeds = load_seeds(SEEDS_FILE)
    logger.info("Всего анонимных seed-диалогов: %d", len(seeds))

    avg_len = sum(len(d) for d in seeds) / len(seeds)
    logger.info("Средняя длина одного окна: %.1f реплик", avg_len)

    try:
        pairs = await process_seeds(seeds)
        logger.info("=" * 60)
        logger.info("Генерация завершена. Всего получено пар: %d", len(pairs))
        split_and_save(pairs)
        logger.info("=" * 60)
    except KeyboardInterrupt:
        logger.warning("Прервано пользователем. Прогресс сохранён в %s", CHECKPOINT_FILE)
        sys.exit(0)
    except Exception as e:
        logger.exception("Критическая ошибка: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())