"""Проверки согласованности документа. Ничего не исправляют — только добавляют warnings.

Правила выбора значения при противоречиях (описаны и в README):
  * две разные итоговые суммы -> total_amount берётся из строки "к оплате" (если её нет — последняя итоговая строка);
  * две разные даты документа -> doc_date = дата в заголовке рядом с номером документа.
Выбор делает модель по правилам из промпта; этот модуль независимо от модели
находит противоречия в тексте и сообщает о них.
"""
from __future__ import annotations

import re

from .normalize import LAT_TO_CYR, NUM_RE, dates_in_text, fix_ocr_digits, match_to_float
from .schema import ExtractionResult

TOTAL_LABEL = re.compile(r"итого|всего|к оплате|жалпы сомасы|барлығы", re.I)
DOC_DATE_LABEL = re.compile(r"№|дата|күні|\bот\b|составлен", re.I)
NOT_DOC_DATE = re.compile(r"действительн|договор|срок|\bдо\b", re.I)
VAT_EXEMPT = re.compile(r"не облагается|без ндс|ққс-сыз|ққс салынбайды|ндс не предусмотрен", re.I)
VAT_INCLUDED = re.compile(r"с уч[её]том ндс|включая ндс|в том числе ндс|в т\.ч\. ндс", re.I)

EPS = 1.0  # допуск на округление, тенге


def _labeled_totals(text: str) -> list[float]:
    """Числа, стоящие после меток 'Итого' / 'Всего' / 'К оплате' / 'Жалпы сомасы'."""
    values = []
    for line in fix_ocr_digits(text).splitlines():
        label = TOTAL_LABEL.search(line.translate(LAT_TO_CYR))  # translate не меняет длину строки
        if not label:
            continue
        number = NUM_RE.search(line, label.end())
        if number:
            values.append(match_to_float(number))
    return values


def check_totals(text: str, result: ExtractionResult) -> list[str]:
    values = sorted(set(_labeled_totals(text)))
    if len(values) < 2:
        return []
    vat = result.vat_amount
    # "Итого 160 500" + "НДС 25 680" = "Всего к оплате 186 180" — это не противоречие.
    if len(values) == 2 and vat is not None and abs(values[0] + vat - values[1]) < EPS:
        return []
    return [f"TOTALS_CONFLICT: в документе разные итоговые суммы {values}; выбрано {result.total_amount} (правило: строка 'к оплате')"]


def check_dates(text: str, result: ExtractionResult) -> list[str]:
    candidates: list[str] = []
    for line in text.splitlines():
        if DOC_DATE_LABEL.search(line) and not NOT_DOC_DATE.search(line):
            candidates += [d for d in dates_in_text(line) if d not in candidates]
    if len(candidates) < 2:
        return []
    return [f"DATES_CONFLICT: в документе несколько дат документа {candidates}; выбрано {result.doc_date} (правило: дата в заголовке)"]


def check_items_sum(result: ExtractionResult) -> list[str]:
    items = result.line_items
    if result.total_amount is None or not items or any(i.qty is None or i.price is None for i in items):
        return []
    s = sum(i.qty * i.price for i in items)
    total, vat = result.total_amount, result.vat_amount or 0.0
    if abs(s - total) < EPS or abs(s + vat - total) < EPS:
        return []
    return [f"ITEMS_SUM_MISMATCH: сумма позиций {s:g} не совпадает с итогом {total:g}"]


def check_vat_notes(text: str, result: ExtractionResult) -> list[str]:
    if result.vat_amount is not None:
        return []
    if VAT_EXEMPT.search(text):
        return ["VAT_EXEMPT: в документе указано, что НДС не облагается; vat_amount = null"]
    if VAT_INCLUDED.search(text):
        return ["VAT_INCLUDED_NOT_STATED: НДС включён в цену, но сумма НДС не указана; не вычисляем, vat_amount = null"]
    return []


def run_checks(text: str, result: ExtractionResult) -> list[str]:
    if result.doc_type == "unknown":
        return []
    return check_totals(text, result) + check_dates(text, result) + check_items_sum(result) + check_vat_notes(text, result)
