"""Пост-обработка: проверка "заземлённости" (grounding) каждого поля в исходном тексте.

Идея: модели нельзя верить на слово. Каждое значение, которое она вернула,
мы ищем в тексте документа детерминированным кодом. Не нашли — поле обнуляется,
а в warnings пишется UNGROUNDED_DROPPED. Так галлюцинация превращается в пропуск,
а пропуск для бизнеса дешевле (его видно, его доберёт человек).

field_confidence считается НЕ моделью (самооценка LLM ненадёжна), а по тому,
как именно значение подтвердилось в тексте:
  0.95 — найдено в тексте как есть;
  0.75 — найдено после нормализации (OCR-исправление, сумма прописью, латиница/кириллица);
  0.5  — найдено, но есть сомнение (не сходится контрольная сумма БИН, тип документа без ключевых слов);
  0.0  — значения нет (null).
"""
from __future__ import annotations

import re
from typing import Any

from .normalize import (
    bin_candidates,
    bin_checksum_ok,
    contains_normalized,
    dates_in_text,
    fix_ocr_digits,
    norm_text,
    number_source,
    to_iso_date,
)
from .schema import SCALAR_FIELDS, ExtractionResult, LineItem, LLMExtraction

CONF_EXACT = 0.95
CONF_NORMALIZED = 0.75
CONF_DOUBTFUL = 0.5
CONF_NONE = 0.0

DOC_TYPE_KEYWORDS = {
    "invoice": ("счет на оплату", "счет-фактура", "счет фактура", "шот-фактура", "шот фактура", "invoice", "счет №"),
    "commercial_offer": ("коммерческое предложение", "коммерческое пред", "кп-", "предлагаем"),
    "act": ("акт выполненных", "акт оказанных", "актісі", "акт №", "акт приема"),
}

CURRENCY_PATTERNS = {
    "KZT": r"тенге|теңге|\bтг\b|₸|\bkzt\b",
    "USD": r"\busd\b|доллар|\$",
    "EUR": r"\beur\b|евро|€",
}

BUYER_WORDS = re.compile(r"заказчик|покупатель|плательщик|получатель|тапсырыс беруші|сатып алушы", re.I)
SUPPLIER_WORDS = re.compile(r"поставщик|исполнитель|продавец|орындаушы|жеткізуші", re.I)


class _Collector:
    """Собирает значения, confidence и предупреждения в одном месте."""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.conf: dict[str, float] = {}
        self.warnings: list[str] = []

    def keep(self, field: str, value: Any, conf: float) -> None:
        self.values[field] = value
        self.conf[field] = conf

    def null(self, field: str) -> None:
        self.values[field] = None
        self.conf[field] = CONF_NONE

    def drop(self, field: str, value: Any, reason: str = "значение не найдено в тексте") -> None:
        self.null(field)
        self.warnings.append(f"UNGROUNDED_DROPPED: {field}={value!r} — {reason}")


def _ground_doc_type(c: _Collector, doc_type: str, text: str) -> None:
    if doc_type == "unknown":
        c.keep("doc_type", "unknown", CONF_NONE)
        return
    # и нормализованный текст (OCR-латиница -> кириллица), и исходный (для английских слов)
    t = norm_text(text) + " " + text.lower().replace("ё", "е")
    if any(kw in t for kw in DOC_TYPE_KEYWORDS[doc_type]):
        c.keep("doc_type", doc_type, CONF_EXACT)
    else:
        c.keep("doc_type", doc_type, CONF_DOUBTFUL)
        c.warnings.append(f"DOC_TYPE_WEAK: нет ключевых слов для типа {doc_type}")


def _ground_name(c: _Collector, name: str | None, text: str) -> None:
    if name is None:
        c.null("supplier_name")
    elif name in text:
        c.keep("supplier_name", name, CONF_EXACT)
    elif contains_normalized(text, name):
        c.keep("supplier_name", name, CONF_NORMALIZED)
    else:
        c.drop("supplier_name", name)


def _bin_role(bin_: str, text: str) -> str | None:
    """Чей БИН: ищем ближайшее слово-роль слева от БИН в той же строке."""
    for line in fix_ocr_digits(text).splitlines():
        pos = line.find(bin_)
        if pos == -1:
            continue
        before = line[:pos]
        roles = [(m.start(), "buyer") for m in BUYER_WORDS.finditer(before)]
        roles += [(m.start(), "supplier") for m in SUPPLIER_WORDS.finditer(before)]
        return max(roles)[1] if roles else None
    return None


def _ground_bin(c: _Collector, value: str | None, text: str) -> None:
    if value is None:
        c.null("supplier_bin")
        return
    normalized = fix_ocr_digits(re.sub(r"[\s-]", "", value))
    candidates = bin_candidates(text)
    if normalized not in candidates:
        c.drop("supplier_bin", value)
        return
    if _bin_role(normalized, text) == "buyer":
        c.drop("supplier_bin", value, "это БИН заказчика/покупателя, а не поставщика")
        return
    conf = CONF_EXACT
    if candidates[normalized]:
        conf = CONF_NORMALIZED
        c.warnings.append(f"BIN_OCR_CORRECTED: БИН в тексте с OCR-ошибками, нормализован в {normalized}")
    if not bin_checksum_ok(normalized):
        conf = CONF_DOUBTFUL
        c.warnings.append(f"BIN_CHECKSUM_FAILED: {normalized} не проходит контрольную сумму, проверьте вручную")
    c.keep("supplier_bin", normalized, conf)


def _ground_doc_number(c: _Collector, value: str | None, text: str) -> None:
    if value is None:
        c.null("doc_number")
        return
    value = re.sub(r"^(№|N°|No\.?|#)\s*", "", value.strip())
    if value and value in text:
        c.keep("doc_number", value, CONF_EXACT)
    elif value and (contains_normalized(text, value) or norm_text(value) in norm_text(fix_ocr_digits(text))):
        c.keep("doc_number", value, CONF_NORMALIZED)
    else:
        c.drop("doc_number", value)


def _ground_date(c: _Collector, value: str | None, text: str) -> None:
    if value is None:
        c.null("doc_date")
        return
    iso = to_iso_date(value)
    if iso is None:
        c.drop("doc_date", value, "не удалось распознать дату")
    elif iso in dates_in_text(text):
        c.keep("doc_date", iso, CONF_EXACT)
    else:
        c.drop("doc_date", value)


def _ground_currency(c: _Collector, value: str | None, text: str) -> None:
    if value is None:
        c.null("currency")
    elif re.search(CURRENCY_PATTERNS[value], text, re.I):
        c.keep("currency", value, CONF_EXACT)
    else:
        c.drop("currency", value, "валюта не названа в тексте")


def _ground_amount(c: _Collector, field: str, value: float | None, text: str) -> None:
    if value is None:
        c.null(field)
        return
    source = number_source(value, text)
    if source == "digits":
        c.keep(field, value, CONF_EXACT)
    elif source == "words":
        c.keep(field, value, CONF_NORMALIZED)
    else:
        c.drop(field, value)


def _name_grounded(name: str, text: str) -> bool:
    """Позиция считается найденной, если >= 70% значимых слов её названия есть в тексте."""
    words = [w for w in re.findall(r"\w+", norm_text(name)) if len(w) >= 3 or w.isdigit()]
    if not words:
        return contains_normalized(text, name)
    t = norm_text(text)
    return sum(w in t for w in words) / len(words) >= 0.7


def _ground_line_items(c: _Collector, items: list[LineItem], text: str) -> None:
    kept: list[LineItem] = []
    for item in items:
        if not _name_grounded(item.name, text):
            c.warnings.append(f"UNGROUNDED_DROPPED: line_item {item.name!r} — позиция не найдена в тексте")
            continue
        qty, price = item.qty, item.price
        if qty is not None and number_source(qty, text) is None:
            c.warnings.append(f"UNGROUNDED_DROPPED: line_item {item.name!r} qty={qty} — не найдено в тексте")
            qty = None
        if price is not None and number_source(price, text) is None:
            c.warnings.append(f"UNGROUNDED_DROPPED: line_item {item.name!r} price={price} — не найдено в тексте")
            price = None
        kept.append(LineItem(name=item.name, qty=qty, price=price))
    c.values["line_items"] = kept


def postprocess(data: LLMExtraction, text: str) -> ExtractionResult:
    c = _Collector()
    c.warnings.extend(f"LLM: {w}" for w in data.warnings if w.strip())

    _ground_doc_type(c, data.doc_type, text)
    _ground_name(c, data.supplier_name, text)
    _ground_bin(c, data.supplier_bin, text)
    _ground_doc_number(c, data.doc_number, text)
    _ground_date(c, data.doc_date, text)
    _ground_currency(c, data.currency, text)
    _ground_amount(c, "total_amount", data.total_amount, text)
    _ground_amount(c, "vat_amount", data.vat_amount, text)
    _ground_line_items(c, data.line_items, text)

    # Правило: если это не счёт/КП/акт, то и "поставщика" с "итогом" у него нет.
    # Реквизиты в письме или договоре могут принадлежать кому угодно — не рискуем.
    if c.values["doc_type"] == "unknown":
        filled = [f for f in SCALAR_FIELDS if f != "doc_type" and c.values.get(f) is not None]
        if filled or c.values["line_items"]:
            c.warnings.append(f"DOC_TYPE_UNKNOWN_FIELDS_CLEARED: тип не распознан, очищены поля {filled}")
        for f in SCALAR_FIELDS:
            if f != "doc_type":
                c.null(f)
        c.values["line_items"] = []

    return ExtractionResult(**c.values, field_confidence=c.conf, warnings=c.warnings)
