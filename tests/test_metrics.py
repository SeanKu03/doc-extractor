from extractor.metrics import classify, match_items, names_match


def test_hallucination_and_miss_are_counted_separately():
    assert classify("supplier_bin", "123456789012", None) == "hallucination"
    assert classify("supplier_bin", None, "123456789012") == "miss"
    assert classify("supplier_bin", "111111111111", "123456789012") == "wrong"
    assert classify("supplier_bin", None, None) == "correct"


def test_doc_type_unknown_counts_as_empty():
    assert classify("doc_type", "invoice", "unknown") == "hallucination"
    assert classify("doc_type", "unknown", "act") == "miss"


def test_alternatives_in_expected():
    assert classify("doc_number", "5O7", ["507", "5O7"]) == "correct"
    assert classify("supplier_name", "ЖШС «Көкжиек Құрылыс»", ['ТОО "Кокжиек Курылыс"', "ЖШС «Көкжиек Құрылыс»"]) == "correct"


def test_names_match_ignores_quotes_legal_form_and_ocr_latin():
    assert names_match('ТОО "Алматы Трейд Компани"', "Алматы Трейд Компани")
    assert names_match('T0O "Бepeкe Агро"', "Береке Агро")
    assert not names_match("Алматы", "Алматы Трейд Компани")


def test_line_items_extra_and_missed():
    exp = [{"name": "Бумага А4", "qty": 50, "price": 2100}]
    pred = [{"name": "Бумага А4", "qty": 50, "price": 2100}, {"name": "Доставка", "qty": 1, "price": 5000}]
    assert match_items(pred, exp) == (1, 1, 0)
    assert classify("line_items", pred, exp) == "wrong"
    assert classify("line_items", [], exp) == "miss"
