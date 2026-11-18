from extractor.checks import run_checks
from extractor.schema import ExtractionResult


def codes(warnings):
    return [w.split(":")[0] for w in warnings]


def result(**kw):
    return ExtractionResult(doc_type="invoice", **kw)


def test_subtotal_plus_vat_is_not_a_conflict():
    text = "Итого: 160 500\nНДС 16%: 25 680\nВсего к оплате: 186 180 тенге"
    assert run_checks(text, result(total_amount=186180, vat_amount=25680)) == []


def test_two_different_totals_without_vat_is_a_conflict():
    text = "Итого: 500 000 тенге\nИтого к оплате: 550 000 тенге"
    assert "TOTALS_CONFLICT" in codes(run_checks(text, result(total_amount=550000)))


def test_two_document_dates_conflict_but_validity_date_is_ignored():
    conflict = "Счёт № 15 от 03.11.2026\nДата выписки счёта: 04.11.2026"
    assert "DATES_CONFLICT" in codes(run_checks(conflict, result(doc_date="2026-11-03")))
    offer = "Предложение действительно до 30.11.2026.\nСоставлено 05.11.2026."
    assert run_checks(offer, result(doc_date="2026-11-05")) == []


def test_vat_exempt_is_reported_and_vat_stays_null():
    assert codes(run_checks("НДС не облагается", result())) == ["VAT_EXEMPT"]
    assert codes(run_checks("ҚҚС-сыз / Без НДС", result())) == ["VAT_EXEMPT"]


def test_items_sum_mismatch():
    r = result(total_amount=550000, line_items=[{"name": "Трубы", "qty": 200, "price": 2500}])
    assert "ITEMS_SUM_MISMATCH" in codes(run_checks("", r))
