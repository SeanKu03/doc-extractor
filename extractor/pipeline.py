"""Пайплайн: LLM -> парсинг JSON -> валидация схемы -> (повтор) -> пост-обработка -> финальная валидация.

Что происходит, если ответ модели не прошёл валидацию:
  1. Ошибки pydantic в кратком виде отправляются модели обратно, просим исправить (до MAX_RETRIES раз).
  2. Если и после повторов ответ невалиден — возвращаем безопасный пустой результат
     (doc_type=unknown, все поля null) с предупреждением LLM_OUTPUT_INVALID.
     Невалидные данные клиенту не отдаются никогда. CLI в этом случае выходит с кодом 1.
"""
from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from .llm import LLMClient
from .postprocess import postprocess
from .prompt import build_messages, build_repair_message
from .schema import ExtractionResult, LLMExtraction, empty_result

log = logging.getLogger(__name__)

MAX_RETRIES = 1
INVALID_WARNING = "LLM_OUTPUT_INVALID"


def summarize_validation_error(err: ValidationError | json.JSONDecodeError) -> str:
    if isinstance(err, json.JSONDecodeError):
        return f"невалидный JSON: {err.msg} (позиция {err.pos})"
    lines = []
    for e in err.errors()[:10]:
        loc = ".".join(str(p) for p in e["loc"]) or "<root>"
        lines.append(f"- {loc}: {e['msg']}")
    return "\n".join(lines)


def parse_llm_output(raw: str) -> LLMExtraction:
    """Строгий парсинг: сначала JSON, потом схема. Любая ошибка — исключение."""
    data = json.loads(raw)
    return LLMExtraction.model_validate(data)


def extract(text: str, llm: LLMClient, max_retries: int = MAX_RETRIES) -> ExtractionResult:
    messages = build_messages(text)
    schema = LLMExtraction.model_json_schema()
    errors: list[str] = []

    for attempt in range(max_retries + 1):
        raw = llm.chat(messages, schema)
        log.debug("attempt %d raw output: %s", attempt, raw)
        try:
            parsed = parse_llm_output(raw)
        except (json.JSONDecodeError, ValidationError) as err:
            summary = summarize_validation_error(err)
            errors.append(f"attempt {attempt + 1}: {summary}")
            log.warning("LLM output invalid (attempt %d): %s", attempt + 1, summary)
            messages = messages + [
                {"role": "assistant", "content": raw},
                build_repair_message(summary),
            ]
            continue

        try:
            result = postprocess(parsed, text)
            if attempt > 0:
                result.warnings.append(f"LLM_OUTPUT_REPAIRED: валидный ответ получен с попытки {attempt + 1}")
            # Финальная проверка: даже наш собственный код не может отдать невалидный объект.
            return ExtractionResult.model_validate(result.model_dump())
        except ValidationError as err:
            log.error("final result failed validation: %s", err)
            return empty_result([f"{INVALID_WARNING}: финальная валидация: {summarize_validation_error(err)}"])

    return empty_result([f"{INVALID_WARNING}: " + " | ".join(errors)])
