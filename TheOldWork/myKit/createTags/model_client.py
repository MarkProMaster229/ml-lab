# model_client.py
import requests
import time
from typing import Dict, Optional
from config import Config

class ModelClient:
    """Клиент для работы с Ollama API"""
    
    def __init__(self, config: Config):
        self.config = config
        self.timeout = 30
    
    def generate_response(self, prompt: str) -> Optional[str]:
        """Генерирует ответ через Ollama API"""
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "context": None,  # Сбрасываем контекст
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.num_predict,
                "repeat_penalty": self.config.repeat_penalty,
                "num_ctx": 4096
            }
        }
        
        try:
            response = requests.post(
                self.config.ollama_url,
                json=payload,
                timeout=self.timeout,
                headers={'Connection': 'close', 'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                print(f"⚠️ Ошибка API: {response.status_code}")
                if response.status_code == 500:
                    print(f"Текст ошибки: {response.text[:200]}")
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
    
    def test_connection(self) -> bool:
        """Тестирует подключение к Ollama"""
        try:
            response = requests.get(f"{self.config.ollama_url.replace('/api/generate', '')}/api/tags", 
                                  timeout=5)
            return response.status_code == 200
        except:
            return False