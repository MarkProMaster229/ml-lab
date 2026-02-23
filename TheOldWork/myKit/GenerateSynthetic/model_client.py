# modules/model_client.py
# -*- coding: utf-8 -*-

import requests
from typing import Optional

from config import Config


class ModelClient:
    """Клиент для работы с Ollama API (или любым совместимым)."""

    def __init__(self, config: Config):
        self.config = config
        self.timeout = 120  # секунд

    # ----------------------------------------------------------------- #
    def generate_response(self, prompt: str) -> Optional[str]:
        """Отправляет запрос к модели и возвращает «чистый» текст ответа."""
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "context": None,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.num_predict,
                "repeat_penalty": self.config.repeat_penalty,
                "num_ctx": 4096,
            },
        }

        try:
            resp = requests.post(
                self.config.ollama_url,
                json=payload,
                timeout=self.timeout,
                headers={"Connection": "close", "Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response", "")
            else:
                print(f"⚠️ Ошибка API: {resp.status_code}")
                if resp.status_code == 500:
                    print(f"Текст ошибки: {resp.text[:200]}")
                return None
        except requests.exceptions.Timeout:
            print("⚠️ Таймаут запроса")
            return None
        except requests.exceptions.ConnectionError:
            print(f"⚠️ Ошибка подключения к Ollama ({self.config.ollama_url})")
            return None
        except Exception as e:
            print(f"⚠️ Неожиданная ошибка: {type(e).__name__}: {e}")
            return None

    # ----------------------------------------------------------------- #
    def test_connection(self) -> bool:
        """Проверка, доступен ли сервер Ollama."""
        try:
            resp = requests.get(
                self.config.ollama_url.replace("/api/generate", "/api/tags"), timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False