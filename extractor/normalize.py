"""Детерминированный разбор текста: числа (цифрами и прописью), даты, БИН, OCR-исправления.

Эти функции не "понимают" документ — они только отвечают на вопрос
"встречается ли это значение в тексте?". На этом строится защита от галлюцинаций.
"""
from __future__ import annotations

import re
from datetime import date

# --- OCR и гомоглифы ----------------------------------------------------------

# Латинские буквы, которые OCR путает с кириллическими (и наоборот).
LAT_TO_CYR = str.maketrans("AaBCcEeHKkMOoPpTXxy", "АаВСсЕеНКкМОоРрТХху")
_OCR_DIGIT_LIKE = {"O": "0", "o": "0", "О": "0", "о": "0", "l": "1", "I": "1", "|": "1", "З": "3"}
_DIGIT_NEIGHBOUR = re.compile(r"(?<=\d)[OoОоlI|З]|[OoОоlI|З](?=\d)")


def fix_ocr_digits(text: str) -> str:
    """Заменяет буквы, похожие на цифры, если они стоят вплотную к цифрам: '2l064090Ol80' -> '210640900180'."""
    prev = None
    while prev != text:
        prev = text
        text = _DIGIT_NEIGHBOUR.sub(lambda m: _OCR_DIGIT_LIKE[m.group(0)], text)
    return text


def norm_text(text: str) -> str:
    """Нормализация для сравнения строк: регистр, ё, кавычки, латиница->кириллица, пробелы."""
    t = text.translate(LAT_TO_CYR).lower().replace("ё", "е")
    t = re.sub(r"[«»\"'“”„`]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def contains_normalized(haystack: str, needle: str) -> bool:
    return norm_text(needle) in norm_text(haystack)


# --- Числа цифрами ------------------------------------------------------------

# "186 180", "42 758,62", "1250", "1 250.50". Разделитель тысяч — один пробел (в т.ч. неразрывный).
NUM_RE = re.compile(r"(?<![\d.,])(\d{1,3}(?:[   ]\d{3})+|\d+)(?:[.,](\d{1,2}))?(?![\d])")


def match_to_float(m: re.Match) -> float:
    integer = re.sub(r"\D", "", m.group(1))
    return float(f"{integer}.{m.group(2)}" if m.group(2) else integer)


def numbers_in_text(text: str) -> set[float]:
    """Числа, записанные цифрами (с учётом OCR-ошибок вида 1O 000)."""
    found: set[float] = set()
    for source in (text, fix_ocr_digits(text)):
        for m in NUM_RE.finditer(source):
            found.add(match_to_float(m))
            # "2 100" может быть и одним числом, и двумя колонками таблицы — сохраняем части тоже.
            for part in re.split(r"[   ]", m.group(1)):
                found.add(float(part))
    return found


# --- Числа прописью (русский и казахский) ------------------------------------

_UNITS = {
    # русский
    "ноль": 0, "один": 1, "одна": 1, "одно": 1, "два": 2, "две": 2, "три": 3, "четыре": 4,
    "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
    "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15,
    "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50, "шестьдесят": 60,
    "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
    "сто": 100, "двести": 200, "триста": 300, "четыреста": 400, "пятьсот": 500,
    "шестьсот": 600, "семьсот": 700, "восемьсот": 800, "девятьсот": 900,
    # казахский
    "бір": 1, "екі": 2, "үш": 3, "төрт": 4, "бес": 5, "алты": 6, "жеті": 7, "сегіз": 8,
    "тоғыз": 9, "он": 10, "жиырма": 20, "отыз": 30, "қырық": 40, "елу": 50, "алпыс": 60,
    "жетпіс": 70, "сексен": 80, "тоқсан": 90,
}
_HUNDRED_MULT = {"жүз"}  # казахское "үш жүз" = 3 * 100
_SCALES = {
    "тысяча": 1_000, "тысячи": 1_000, "тысяч": 1_000, "мың": 1_000,
    "миллион": 1_000_000, "миллиона": 1_000_000, "миллионов": 1_000_000,
    "миллиард": 1_000_000_000, "миллиарда": 1_000_000_000, "миллиардов": 1_000_000_000,
}


def _parse_number_words(tokens: list[str]) -> int:
    total, current = 0, 0
    for tok in tokens:
        if tok in _UNITS:
            current += _UNITS[tok]
        elif tok in _HUNDRED_MULT:
            current = max(current, 1) * 100
        elif tok in _SCALES:
            total += max(current, 1) * _SCALES[tok]
            current = 0
    return total + current


def words_numbers_in_text(text: str) -> set[float]:
    """Находит все последовательности числительных и переводит их в числа."""
    tokens = re.findall(r"[a-zа-яёәғқңөұүһі]+", text.lower().replace("ё", "е"))
    result: set[float] = set()
    run: list[str] = []
    for tok in tokens + [""]:  # пустой токен закрывает последнюю последовательность
        if tok in _UNITS or tok in _HUNDRED_MULT or tok in _SCALES:
            run.append(tok)
        elif run:
            # одиночное "он"/"бес"/"сто" — слишком двусмысленно, чтобы считать числом
            if len(run) > 1:
                result.add(float(_parse_number_words(run)))
            run = []
    return result


def number_source(value: float, text: str) -> str | None:
    """Где число встречается в тексте: 'digits', 'words' или None (не встречается)."""
    if any(abs(value - n) < 0.005 for n in numbers_in_text(text)):
        return "digits"
    if any(abs(value - n) < 0.005 for n in words_numbers_in_text(text)):
        return "words"
    return None


# --- Даты ---------------------------------------------------------------------

_MONTHS = {
    # русский (родительный падеж)
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6, "июля": 7,
    "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
    # казахский
    "қаңтар": 1, "ақпан": 2, "наурыз": 3, "сәуір": 4, "мамыр": 5, "маусым": 6, "шілде": 7,
    "тамыз": 8, "қыркүйек": 9, "қазан": 10, "қараша": 11, "желтоқсан": 12,
}
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b")
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_WORD_DATE = re.compile(r"\b(\d{1,2})\s+([а-яәғқңөұүһі]+)\s+(\d{4})", re.IGNORECASE)


def _safe_date(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def dates_in_text(text: str) -> list[str]:
    """Все даты из текста в ISO-формате, в порядке появления, без повторов."""
    text = fix_ocr_digits(text)
    found: list[tuple[int, str]] = []
    for m in _NUMERIC_DATE.finditer(text):
        iso = _safe_date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        if iso:
            found.append((m.start(), iso))
    for m in _ISO_DATE.finditer(text):
        iso = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if iso:
            found.append((m.start(), iso))
    for m in _WORD_DATE.finditer(text):
        month = _MONTHS.get(m.group(2).lower())
        if month:
            iso = _safe_date(int(m.group(3)), month, int(m.group(1)))
            if iso:
                found.append((m.start(), iso))
    seen, ordered = set(), []
    for _, iso in sorted(found):
        if iso not in seen:
            seen.add(iso)
            ordered.append(iso)
    return ordered


def to_iso_date(value: str) -> str | None:
    """Приводит дату, которую вернула модель, к YYYY-MM-DD (модель может вернуть 12.11.2026)."""
    value = value.strip()
    if _ISO_DATE.fullmatch(value):
        y, m, d = value.split("-")
        return _safe_date(int(y), int(m), int(d))
    dates = dates_in_text(value)
    return dates[0] if len(dates) == 1 else None


# --- БИН ----------------------------------------------------------------------

_BIN_TOKEN = re.compile(r"(?<![\w])[\dOoОоlI|З]{12}(?![\w])")


def bin_checksum_ok(bin_: str) -> bool:
    """Контрольный разряд ИИН/БИН РК (12-я цифра)."""
    if not re.fullmatch(r"\d{12}", bin_):
        return False
    d = [int(c) for c in bin_]
    s = sum(d[i] * (i + 1) for i in range(11)) % 11
    if s == 10:
        weights = [3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2]
        s = sum(d[i] * weights[i] for i in range(11)) % 11
        if s == 10:
            return False
    return s == d[11]


def bin_candidates(text: str) -> dict[str, bool]:
    """Все 12-значные токены из текста: {нормализованный_БИН: был_ли_исправлен_OCR}."""
    result: dict[str, bool] = {}
    for m in _BIN_TOKEN.finditer(text):
        raw = m.group(0)
        if sum(c.isdigit() for c in raw) < 8:  # слишком мало цифр — это не БИН
            continue
        fixed = fix_ocr_digits(raw)
        if re.fullmatch(r"\d{12}", fixed):
            # если тот же БИН где-то встречается без ошибок — считаем его неисправленным
            result[fixed] = result.get(fixed, True) and fixed != raw
    return result
