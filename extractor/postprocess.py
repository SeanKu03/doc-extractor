"""Пост-обработка ответа модели. Пока — только приведение к финальной схеме."""
from __future__ import annotations

from .schema import SCALAR_FIELDS, ExtractionResult, LLMExtraction


def postprocess(data: LLMExtraction, text: str) -> ExtractionResult:
    values = data.model_dump(exclude={"warnings"})
    confidence = {f: (0.0 if values.get(f) is None else 0.5) for f in SCALAR_FIELDS}
    return ExtractionResult(**values, field_confidence=confidence, warnings=list(data.warnings))
