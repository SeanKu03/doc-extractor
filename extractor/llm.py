"""Клиенты LLM.

OllamaClient  — настоящий вызов локальной модели через HTTP API Ollama.
ScriptedLLM   — заглушка для тестов: отдаёт заранее заданные ответы по очереди.
                В eval не используется — eval всегда гоняет настоящую модель.
"""
from __future__ import annotations

import json
import os
from typing import Protocol

import requests


class LLMUnavailable(RuntimeError):
    """Модель недоступна (Ollama не запущена, модель не скачана, таймаут)."""


class LLMClient(Protocol):
    name: str

    def chat(self, messages: list[dict], json_schema: dict) -> str:
        """Вернуть сырой текст ответа модели (ожидаем JSON, но не доверяем этому)."""
        ...


class OllamaClient:
    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma3:4b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self.name = f"ollama:{self.model}"

    def chat(self, messages: list[dict], json_schema: dict) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # Structured outputs: Ollama ограничивает генерацию JSON-схемой.
            # Это снижает число битых ответов, но НЕ заменяет нашу валидацию.
            "format": json_schema,
            "options": {"temperature": 0, "num_ctx": 8192},
        }
        try:
            resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            raise LLMUnavailable(
                f"Не удалось обратиться к Ollama на {self.host}: {e}. "
                f"Запущена ли Ollama (`ollama serve`)?"
            ) from e
        if resp.status_code == 404:
            raise LLMUnavailable(
                f"Модель {self.model} не найдена. Выполните `ollama pull {self.model}`."
            )
        if resp.status_code != 200:
            raise LLMUnavailable(f"Ollama вернула {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()["message"]["content"]
        except (ValueError, KeyError) as e:
            raise LLMUnavailable(f"Неожиданный формат ответа Ollama: {resp.text[:300]}") from e


class ScriptedLLM:
    """Тестовая заглушка: возвращает ответы из списка по порядку и запоминает запросы."""

    name = "scripted"

    def __init__(self, responses: list[str | dict]) -> None:
        self._responses = [r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in responses]
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], json_schema: dict) -> str:
        self.calls.append([dict(m) for m in messages])
        if not self._responses:
            raise AssertionError("ScriptedLLM: ответы закончились")
        return self._responses.pop(0)
