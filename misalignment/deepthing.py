import ollama
import time
import json
import re
from typing import List, Optional, Dict, Any

class AutonomousThinker:
    def __init__(
        self,
        model_name: str,
        system_prompt: str = "",
        initial_context: str = "",
        temperature: float = 0.8,
        top_p: float = 0.9,
        num_ctx: int = 4096,
        stop_tokens: Optional[List[str]] = None,
        use_thinking: bool = False
    ):
        self.model = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.num_ctx = num_ctx
        self.stop_tokens = stop_tokens or []
        self.use_thinking = use_thinking

        if isinstance(initial_context, list):
            self.context = self._messages_to_text(initial_context)
        else:
            self.context = initial_context

        self.history: List[Dict[str, Any]] = []

    @staticmethod
    def _messages_to_text(messages: List[Dict[str, str]]) -> str:
        lines = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                lines.append(f"System: {content}")
            elif role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
        return "\n".join(lines)

    def _prepare_prompt(self) -> str:
        """
        Возвращает ТОЛЬКО system_prompt и накопленный контекст.
        НИКАКИХ дополнительных инструкций.
        Если оба пустые — будет пустая строка.
        """
        parts = []
        if self.system_prompt.strip():
            parts.append(self.system_prompt.strip())
        if self.context.strip():
            parts.append(self.context.strip())
        return "\n\n".join(parts)

    def _truncate_context(self, text: str, max_chars: int = 30000) -> str:
        if len(text) > max_chars:
            return text[-max_chars:]
        return text

    @staticmethod
    def _extract_thinking(text: str) -> tuple:
        """
        Извлекает <think>...</think> из текста.
        Возвращает (thinking, thought).
        """
        thinking = ""
        thought = text.strip()

        # Пробуем найти <think>...</think>
        match = re.search(r"<think>(.*?)</think>", thought, re.DOTALL | re.IGNORECASE)
        if match:
            thinking = match.group(1).strip()
            thought = thought.replace(match.group(0), "").strip()

        return thinking, thought

    def step(self) -> Dict[str, Any]:
        """
        Один шаг. Возвращает словарь с thinking и thought.
        """
        prompt = self._prepare_prompt()

        # Если prompt пустой, передаём действительно пустую строку
        # (Ollama может ругаться, поэтому на всякий случай один пробел)
        if not prompt.strip():
            prompt = ""  # или " " если нужен хоть один символ

        options = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_ctx": self.num_ctx,
            "num_predict": -1,
            "stop": self.stop_tokens,
        }

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options=options
        )

        raw_response = response.get("response", "") or ""
        if not isinstance(raw_response, str):
            raw_response = str(raw_response)

        thinking, thought = self._extract_thinking(raw_response)

        record = {
            "step": len(self.history) + 1,
            "prompt": prompt,  # сохраняем сам prompt
            "prompt_length_chars": len(prompt),
            "thinking": thinking,
            "thought": thought,
            "timestamp": time.time()
        }
        self.history.append(record)

        if thought:
            self.context = (self.context + "\n\n" + thought).strip()
            self.context = self._truncate_context(self.context)

        return record

    def run(self, max_steps: Optional[int] = None,
            verbose: bool = True,
            save_to_file: Optional[str] = None,
            stop_on_empty: bool = False,
            max_empty_steps: int = 3) -> List[Dict[str, Any]]:
        empty_streak = 0
        step_count = 0

        try:
            while True:
                if max_steps is not None and step_count >= max_steps:
                    break

                record = self.step()
                step_count += 1

                if verbose:
                    print(f"--- Step {step_count} ---")
                    if record["thinking"]:
                        print("*** THINKING ***")
                        print(record["thinking"])
                        print("*** ANSWER ***")
                    print(record["thought"])
                    print()

                if save_to_file:
                    self.save_history(save_to_file)

                if stop_on_empty and not record["thought"]:
                    empty_streak += 1
                    if empty_streak >= max_empty_steps:
                        print("Несколько пустых шагов подряд. Останавливаюсь.")
                        break
                else:
                    empty_streak = 0

        except KeyboardInterrupt:
            print("\nПрервано пользователем (Ctrl+C).")
        except Exception as e:
            print(f"Ошибка: {e}")
        finally:
            if save_to_file:
                self.save_history(save_to_file)
            return self.history

    def save_history(self, filename: str):
        """Сохраняет историю в JSON-файл."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def reset(self, new_context: str = ""):
        self.context = new_context
        self.history = []


# Пример использования
if __name__ == "__main__":
    thinker = AutonomousThinker(
        model_name="gemma4:26b",
        system_prompt="𐤟ⶳ𓃰𝔛Ꙁ⨠冬𝄡",
        initial_context="",
        temperature=0.9,
        top_p=0.95,
        num_ctx=4096,
        use_thinking=False
    )

    thinker.run(
        max_steps=None,
        verbose=True,
        save_to_file="monolog.json",
        stop_on_empty=False
    )