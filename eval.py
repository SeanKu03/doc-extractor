"""Мини-eval: прогоняет все документы из data/docs через настоящую модель и сравнивает с data/expected.

    python eval.py                       # модель из $OLLAMA_MODEL или qwen2.5:7b
    python eval.py --model llama3.1:8b
    python eval.py --only 03             # только документы, имя которых начинается с 03

Предсказания сохраняются в out/predictions/, сводный отчёт — в out/eval_report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from extractor.llm import LLMUnavailable, OllamaClient
from extractor.metrics import CATEGORIES, FIELDS, classify, match_items
from extractor.pipeline import INVALID_WARNING, extract

ROOT = Path(__file__).parent


def load_dataset(data_dir: Path, only: str | None) -> list[tuple[str, str, dict]]:
    items = []
    for doc_path in sorted((data_dir / "docs").glob("*.txt")):
        if only and not doc_path.stem.startswith(only):
            continue
        exp_path = data_dir / "expected" / f"{doc_path.stem}.json"
        if not exp_path.exists():
            print(f"[skip] нет эталона для {doc_path.name}", file=sys.stderr)
            continue
        items.append((doc_path.stem, doc_path.read_text(encoding="utf-8"), json.loads(exp_path.read_text(encoding="utf-8"))))
    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--only", help="префикс имени документа")
    args = parser.parse_args()

    dataset = load_dataset(Path(args.data_dir), args.only)
    if not dataset:
        print("Нет документов для оценки", file=sys.stderr)
        return 2

    llm = OllamaClient(model=args.model)
    out_dir = ROOT / "out" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Counter] = {f: Counter() for f in FIELDS}
    errors: list[str] = []
    item_totals = Counter()
    warn_found = warn_expected = 0
    invalid = repaired = dropped = 0
    started = time.time()

    for name, text, expected in dataset:
        t0 = time.time()
        try:
            result = extract(text, llm)
        except LLMUnavailable as e:
            print(f"LLM недоступна: {e}", file=sys.stderr)
            return 2
        pred = result.model_dump()
        (out_dir / f"{name}.json").write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")

        invalid += any(w.startswith(INVALID_WARNING) for w in result.warnings)
        repaired += any(w.startswith("LLM_OUTPUT_REPAIRED") for w in result.warnings)
        dropped += sum(w.startswith("UNGROUNDED_DROPPED") for w in result.warnings)

        for field in FIELDS:
            category = classify(field, pred[field], expected[field])
            stats[field][category] += 1
            if category != "correct":
                errors.append(f"  {name:32} {field:14} {category:13} got={pred[field]!r} expected={expected[field]!r}")

        m, extra, missed = match_items(pred["line_items"], expected["line_items"])
        item_totals.update(matched=m, extra=extra, missed=missed, expected=len(expected["line_items"]))

        codes = {w.split(":")[0] for w in result.warnings}
        for code in expected.get("expected_warnings", []):
            warn_expected += 1
            if code in codes:
                warn_found += 1
            else:
                errors.append(f"  {name:32} {'warnings':14} {'not_flagged':13} expected code {code}")
        print(f"[{time.time() - t0:5.1f}s] {name}", file=sys.stderr)

    n = len(dataset)
    elapsed = time.time() - started
    halluc = sum(s["hallucination"] for s in stats.values())
    miss = sum(s["miss"] for s in stats.values())
    wrong = sum(s["wrong"] for s in stats.values())

    print(f"\nМодель: {llm.name}   документов: {n}   время: {elapsed:.0f}s ({elapsed / n:.1f}s/док)\n")
    print(f"{'поле':14} {'точность':>9} {'верно':>6} {'неверно':>8} {'галлюц.':>8} {'пропуск':>8}")
    for field in FIELDS:
        s = stats[field]
        print(f"{field:14} {s['correct'] / n:9.0%} {s['correct']:6} {s['wrong']:8} {s['hallucination']:8} {s['miss']:8}")
    total_cells = n * len(FIELDS)
    correct = sum(s["correct"] for s in stats.values())
    print(f"{'ИТОГО':14} {correct / total_cells:9.0%} {correct:6} {wrong:8} {halluc:8} {miss:8}")

    print(f"\nГаллюцинации (заполнено, а в эталоне null): {halluc}")
    print(f"Пропуски (null, а в эталоне есть значение):  {miss}")
    print(f"Неверные значения (оба заполнены, не равны): {wrong}")
    print(
        f"Позиции: совпало {item_totals['matched']}/{item_totals['expected']}, "
        f"лишних (выдуманных) {item_totals['extra']}, пропущено {item_totals['missed']}"
    )
    print(f"Ожидаемые предупреждения найдены: {warn_found}/{warn_expected}")
    print(f"Ответов модели, не прошедших валидацию: {invalid} (исправлено повтором: {repaired})")
    print(f"Значений модели, отброшенных grounding-проверкой (не найдены в тексте): {dropped}")
    if errors:
        print("\nРасхождения:")
        print("\n".join(errors))

    report = {
        "model": llm.name,
        "documents": n,
        "seconds": round(elapsed, 1),
        "per_field": {f: dict(stats[f]) | {"accuracy": round(stats[f]["correct"] / n, 3)} for f in FIELDS},
        "hallucinations": halluc,
        "misses": miss,
        "wrong": wrong,
        "line_items": dict(item_totals),
        "warnings_found": f"{warn_found}/{warn_expected}",
        "invalid_outputs": invalid,
        "repaired_outputs": repaired,
        "ungrounded_dropped": dropped,
        "errors": [e.strip() for e in errors],
    }
    (ROOT / "out" / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
