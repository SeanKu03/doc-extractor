"""CLI: python -m extractor <file.txt>   (или текст через stdin)

Коды выхода:
  0 — успех;
  1 — модель так и не вернула валидный ответ (выдан безопасный пустой результат);
  2 — модель недоступна / ошибка ввода.
"""
from __future__ import annotations

import argparse
import logging
import sys

from .llm import LLMUnavailable, OllamaClient
from .pipeline import INVALID_WARNING, extract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Извлечь JSON из текста документа")
    parser.add_argument("file", nargs="?", help="путь к .txt; если не указан — читаем stdin")
    parser.add_argument("--model", help="модель Ollama (по умолчанию $OLLAMA_MODEL или qwen2.5:7b)")
    parser.add_argument("-v", "--verbose", action="store_true", help="печатать сырые ответы модели в stderr")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, stream=sys.stderr)

    try:
        if args.file:
            with open(args.file, encoding="utf-8") as f:
                text = f.read()
        else:
            text = sys.stdin.read()
    except OSError as e:
        print(f"Ошибка чтения: {e}", file=sys.stderr)
        return 2
    if not text.strip():
        print("Пустой документ", file=sys.stderr)
        return 2

    try:
        result = extract(text, OllamaClient(model=args.model))
    except LLMUnavailable as e:
        print(f"LLM недоступна: {e}", file=sys.stderr)
        return 2

    print(result.model_dump_json(indent=2))
    return 1 if any(w.startswith(INVALID_WARNING) for w in result.warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
