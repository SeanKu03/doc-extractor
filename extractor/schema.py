"""Схемы данных.

Два уровня валидации:
  * LLMExtraction  — то, что мы требуем от модели. Структура жёсткая (типы, enum,
    запрет лишних полей), но строковые форматы (БИН, дата) мягкие: модель может
    вернуть "2l064090Ol80" или "09.10.2026", и это нормализует наш код, а не модель.
  * ExtractionResult — финальный выход сервиса. Здесь проверяется всё:
    БИН ровно 12 цифр, дата — реальная дата в формате YYYY-MM-DD, суммы >= 0,
    confidence в [0, 1]. Если наш собственный код соберёт невалидный ответ,
    это тоже упадёт, а не уйдёт клиенту.
"""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

DocType = Literal["invoice", "commercial_offer", "act", "unknown"]
Currency = Literal["KZT", "USD", "EUR"]

SCALAR_FIELDS = (
    "doc_type",
    "supplier_name",
    "supplier_bin",
    "doc_number",
    "doc_date",
    "currency",
    "total_amount",
    "vat_amount",
)


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    qty: Optional[float] = Field(default=None, ge=0)
    price: Optional[float] = Field(default=None, ge=0)


class LLMExtraction(BaseModel):
    """Сырой ответ модели после парсинга JSON."""

    model_config = ConfigDict(extra="forbid")

    doc_type: DocType
    supplier_name: Optional[str] = None
    supplier_bin: Optional[str] = None
    doc_number: Optional[str] = None
    doc_date: Optional[str] = None
    currency: Optional[Currency] = None
    total_amount: Optional[float] = Field(default=None, ge=0)
    vat_amount: Optional[float] = Field(default=None, ge=0)
    line_items: list[LineItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("supplier_name", "supplier_bin", "doc_number", "doc_date", mode="before")
    @classmethod
    def empty_string_is_null(cls, v):
        # Модели любят отвечать "" или "null" вместо null.
        if isinstance(v, str) and v.strip().lower() in {"", "null", "none", "n/a", "-"}:
            return None
        return v


class ExtractionResult(BaseModel):
    """Финальный ответ сервиса — строго по схеме из ТЗ."""

    model_config = ConfigDict(extra="forbid")

    doc_type: DocType
    supplier_name: Optional[str] = None
    supplier_bin: Optional[str] = Field(default=None, pattern=r"^\d{12}$")
    doc_number: Optional[str] = None
    doc_date: Optional[str] = None
    currency: Optional[Currency] = None
    total_amount: Optional[float] = Field(default=None, ge=0)
    vat_amount: Optional[float] = Field(default=None, ge=0)
    line_items: list[LineItem] = Field(default_factory=list)
    field_confidence: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("doc_date")
    @classmethod
    def iso_date(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        date.fromisoformat(v)  # бросит ValueError на "2026-13-45" или "12.11.2026"
        if len(v) != 10:
            raise ValueError("doc_date must be YYYY-MM-DD")
        return v

    @field_validator("field_confidence")
    @classmethod
    def confidence_range(cls, v: dict[str, float]) -> dict[str, float]:
        for key, value in v.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"confidence for {key} out of [0,1]: {value}")
        return v


def empty_result(warnings: list[str]) -> ExtractionResult:
    """Безопасный ответ, когда извлечь ничего достоверного не удалось."""
    return ExtractionResult(
        doc_type="unknown",
        field_confidence={f: 0.0 for f in SCALAR_FIELDS},
        warnings=warnings,
    )
