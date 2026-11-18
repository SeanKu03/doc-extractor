"""Сравнение предсказания с эталоном. Отдельно от eval.py, чтобы покрыть тестами.

Каждое поле каждого документа попадает ровно в одну категорию:
  correct        — совпало (в т.ч. оба пустые);
  hallucination  — в эталоне null, а модель заполнила (самая дорогая ошибка);
  miss           — в эталоне есть значение, а у нас null (дешевле: видно, что поля нет);
  wrong          — оба заполнены, но не совпадают.
"""
from __future__ import annotations

import re
from typing import Any

from .normalize import fix_ocr_digits, norm_text
from .schema import SCALAR_FIELDS

FIELDS = SCALAR_FIELDS + ("line_items",)
CATEGORIES = ("correct", "wrong", "hallucination", "miss")

_LEGAL_FORM = re.compile(r"^(тоо|т0о|жшс|ип|ао|llp)\s+")


def _name_norm(s: str) -> str:
    return _LEGAL_FORM.sub("", norm_text(s)).strip()


def names_match(pred: str, exp: str) -> bool:
    """Мягкое сравнение названий: без кавычек, формы собственности, регистра, OCR-латиницы."""
    a, b = _name_norm(pred), _name_norm(exp)
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    return bool(short) and short in long_ and len(short) >= 0.6 * len(long_)


def is_empty(field: str, value: Any) -> bool:
    if field == "doc_type":
        return value in (None, "unknown")
    if field == "line_items":
        return not value
    return value is None


def _alternatives(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _num_eq(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) < 0.01


def _item_match(pred: dict, exp: dict) -> bool:
    return (
        any(names_match(pred["name"], n) for n in _alternatives(exp["name"]))
        and _num_eq(pred.get("qty"), exp.get("qty"))
        and _num_eq(pred.get("price"), exp.get("price"))
    )


def match_items(pred: list[dict], exp: list[dict]) -> tuple[int, int, int]:
    """Возвращает (совпало, лишних у модели, пропущено из эталона)."""
    unused = list(pred)
    matched = 0
    for e in exp:
        for p in unused:
            if _item_match(p, e):
                unused.remove(p)
                matched += 1
                break
    return matched, len(unused), len(exp) - matched


def values_match(field: str, pred: Any, exp: Any) -> bool:
    if field == "line_items":
        m, extra, missed = match_items(pred, exp)
        return extra == 0 and missed == 0
    if field in ("total_amount", "vat_amount"):
        return _num_eq(pred, exp)
    if field == "supplier_name":
        return any(names_match(pred, e) for e in _alternatives(exp))
    if field == "doc_number":
        return any(norm_text(fix_ocr_digits(pred)) == norm_text(fix_ocr_digits(e)) for e in _alternatives(exp))
    return any(pred == e for e in _alternatives(exp))


def classify(field: str, pred: Any, exp: Any) -> str:
    pred_empty, exp_empty = is_empty(field, pred), is_empty(field, exp)
    if pred_empty and exp_empty:
        return "correct"
    if exp_empty:
        return "hallucination"
    if pred_empty:
        return "miss"
    return "correct" if values_match(field, pred, exp) else "wrong"
