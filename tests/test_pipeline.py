"""Пайплайн целиком, с подменённой моделью (ScriptedLLM). Проверяем поведение системы, а не модели."""
import pytest
from pydantic import ValidationError

from extractor.llm import ScriptedLLM
from extractor.pipeline import INVALID_WARNING, extract
from extractor.schema import ExtractionResult

DOC1 = """ТОО "Алматы Трейд Компани"
БИН 170340021554
Счёт на оплату № 1245 от 12.11.2026
Бумага А4 80г/м2     50     2 100     105 000
Итого: 105 000
НДС 16%: 16 800
Всего к оплате: 121 800 тенге
"""

GOOD = {
    "doc_type": "invoice",
    "supplier_name": 'ТОО "Алматы Трейд Компани"',
    "supplier_bin": "170340021554",
    "doc_number": "1245",
    "doc_date": "2026-11-12",
    "currency": "KZT",
    "total_amount": 121800,
    "vat_amount": 16800,
    "line_items": [{"name": "Бумага А4 80г/м2", "qty": 50, "price": 2100}],
    "warnings": [],
}


def codes(result):
    return [w.split(":")[0] for w in result.warnings]


def test_valid_answer_passes_through():
    r = extract(DOC1, ScriptedLLM([GOOD]))
    assert r.total_amount == 121800 and r.doc_date == "2026-11-12"
    assert r.field_confidence["total_amount"] == 0.95


def test_invalid_json_is_repaired_by_retry_with_error_feedback():
    llm = ScriptedLLM(["Вот ваш JSON: {doc_type: invoice", GOOD])
    r = extract(DOC1, llm)
    assert r.doc_type == "invoice"
    assert "LLM_OUTPUT_REPAIRED" in codes(r)
    # во втором запросе модель получила текст ошибки
    assert "не прошёл валидацию" in llm.calls[1][-1]["content"]


def test_schema_violation_twice_returns_safe_empty_result():
    bad = dict(GOOD, doc_type="bill", total_amount="сто")
    r = extract(DOC1, ScriptedLLM([bad, bad]))
    assert r.doc_type == "unknown"
    assert r.total_amount is None and r.supplier_bin is None and r.line_items == []
    assert codes(r) == [INVALID_WARNING]


def test_extra_fields_from_model_are_rejected():
    r = extract(DOC1, ScriptedLLM([dict(GOOD, iban="KZ123"), dict(GOOD, iban="KZ123")]))
    assert codes(r) == [INVALID_WARNING]


@pytest.mark.parametrize(
    "field, hallucinated",
    [
        ("supplier_bin", "123456789012"),      # БИН, которого нет в тексте
        ("doc_date", "2026-11-18"),            # "сегодняшняя" дата
        ("vat_amount", 14_617.0),              # НДС, посчитанный моделью
        ("supplier_name", "ТОО Рога и Копыта"),
    ],
)
def test_hallucinated_value_is_dropped_to_null(field, hallucinated):
    r = extract(DOC1, ScriptedLLM([dict(GOOD, **{field: hallucinated})]))
    assert getattr(r, field) is None
    assert r.field_confidence[field] == 0.0
    assert any(w.startswith(f"UNGROUNDED_DROPPED: {field}") for w in r.warnings)


def test_buyer_bin_is_not_accepted_as_supplier_bin():
    doc = "ТОО «Сервис Плюс»\nАкт № 34 от 31.10.2026\nЗаказчик: ТОО «Алем», БИН 080640029142\nВсего: 1 000 тенге"
    r = extract(doc, ScriptedLLM([{"doc_type": "act", "supplier_bin": "080640029142"}]))
    assert r.supplier_bin is None


def test_unknown_document_gets_no_supplier_fields():
    letter = "Просим выставить счёт. Наши реквизиты: ТОО «Жетысу Офис», БИН 160940018155. Сумма около 85 000 тенге."
    answer = {"doc_type": "unknown", "supplier_name": "ТОО «Жетысу Офис»", "supplier_bin": "160940018155", "total_amount": 85000}
    r = extract(letter, ScriptedLLM([answer]))
    assert r.supplier_name is None and r.supplier_bin is None and r.total_amount is None


def test_ocr_bin_is_normalized_and_flagged():
    doc = "ТОО Кокжиек\nБСН/БИН: 2l064090Ol80\nШот-фактура № 9\nЖалпы сомасы: 340 000 теңге"
    r = extract(doc, ScriptedLLM([{"doc_type": "invoice", "supplier_bin": "2l064090Ol80"}]))
    assert r.supplier_bin == "210640900180"
    assert r.field_confidence["supplier_bin"] == 0.75
    assert "BIN_OCR_CORRECTED" in codes(r)


def test_final_schema_rejects_malformed_values():
    with pytest.raises(ValidationError):
        ExtractionResult(doc_type="invoice", supplier_bin="12345")
    with pytest.raises(ValidationError):
        ExtractionResult(doc_type="invoice", doc_date="12.11.2026")
    with pytest.raises(ValidationError):
        ExtractionResult(doc_type="invoice", field_confidence={"supplier_bin": 1.5})
