import pytest

from extractor.normalize import (
    bin_candidates,
    bin_checksum_ok,
    dates_in_text,
    fix_ocr_digits,
    number_source,
    numbers_in_text,
    to_iso_date,
    words_numbers_in_text,
)


@pytest.mark.parametrize(
    "text, value",
    [
        ("Жалпы сомасы: Үш жүз қырық мың теңге 00 тиын", 340_000),
        ("Итого к оплате: Две тысячи пятьсот долларов США", 2_500),
        ("Сумма: один миллион двести тысяч триста сорок пять тенге", 1_200_345),
        ("Барлығы: екі мың бес жүз теңге", 2_500),
    ],
)
def test_amount_in_words_ru_kz(text, value):
    assert value in words_numbers_in_text(text)


def test_single_number_word_is_not_a_number():
    # "он" (казахское "10") и "сто" как одиночные слова слишком двусмысленны
    assert words_numbers_in_text("он пришёл, сто раз говорил") == set()


def test_numbers_with_thousand_separators_and_decimals():
    nums = numbers_in_text("Итого: 186 180 тенге, НДС 42 758,62")
    assert 186180 in nums and 42758.62 in nums


def test_number_source_distinguishes_digits_and_words():
    assert number_source(340000, "340 000") == "digits"
    assert number_source(340000, "Үш жүз қырық мың") == "words"
    assert number_source(25680, "Итого 160 500") is None


def test_ocr_digits_fixed_only_next_to_digits():
    assert fix_ocr_digits("БИН: 2l064090Ol80") == "БИН: 210640900180"
    assert fix_ocr_digits("ТОО Олжас") == "ТОО Олжас"  # буквы в словах не трогаем


def test_bin_candidates_marks_ocr_correction():
    assert bin_candidates("БСН/БИН: 2l064090Ol80") == {"210640900180": True}
    assert bin_candidates("БИН 170340021554") == {"170340021554": False}


def test_bin_checksum():
    assert bin_checksum_ok("210640900180")
    assert not bin_checksum_ok("170340021554")  # БИН из документа 1 ТЗ не проходит контроль
    assert not bin_checksum_ok("12345")


def test_dates_ru_kz_numeric_and_ocr():
    assert dates_in_text("Шот-фактура № AKT-09 / 09 қазан 2026 ж.") == ["2026-10-09"]
    assert dates_in_text("Счёт № 1 от 2 ноября 2026 г.") == ["2026-11-02"]
    assert dates_in_text("oт 1O.11.2026") == ["2026-11-10"]
    assert dates_in_text("31.02.2026") == []  # несуществующая дата


def test_to_iso_date():
    assert to_iso_date("2026-11-12") == "2026-11-12"
    assert to_iso_date("12.11.2026") == "2026-11-12"
    assert to_iso_date("вчера") is None
