#!/usr/bin/env python3
"""A/B описаний инструментов: доля верного выбора и интервал Вильсона.

Задание 1, пункт 3: два варианта описаний одних и тех же инструментов — «короткие»
(одна фраза) и «полные» (что делает / когда / когда НЕ / единицы / пример). Метрика —
доля задач, где выбран верный инструмент И верные аргументы. Интервал Вильсона.

Скрипт читает журналы runs/*.jsonl (их пишет check.py), группирует по варианту
short/full, для каждой траектории проверяет первый вызов инструмента, считает долю
верного выбора и 95% интервал Вильсона, и делает вывод: окупились ли лишние токены.

Запуск (из week2/practice):
    uv run python ab_report.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

# Чтобы `from check import wilson_interval` работал из любого каталога, добавляем
# каталог этого файла в начало sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check import wilson_interval

RUNS_DIR = Path(__file__).with_name("runs")


# ────────────────────────────────────────────────────────────────────────────
# ОЖИДАЕМЫЙ ВЫЗОВ ДЛЯ КАЖДОЙ ЗАДАЧИ
# ────────────────────────────────────────────────────────────────────────────
# Для каждой задачи известен правильный инструмент и ключевые аргументы.
# «Верно» = первый вызванный инструмент тот же И ключевые аргументы совпали.
# Ключ задачи — её текст (он уникален и совпадает с question в траектории).
EXPECTED: dict[str, dict] = {
    "Сколько в Ростове круглосуточных аптек?": {
        "tool": "count_places", "args": {"kind": "pharmacy", "only_24_7": True}},
    "Сколько всего аптек в Ростове?": {
        "tool": "count_places", "args": {"kind": "pharmacy"}},
    "Сколько супермаркетов в городе?": {
        "tool": "count_places", "args": {"kind": "supermarket"}},
    "Сколько школ в Ростове?": {
        "tool": "count_places", "args": {"kind": "school"}},
    "Сколько банков в городе?": {
        "tool": "count_places", "args": {"kind": "bank"}},
    "Сколько кафе в Ростове?": {
        "tool": "count_places", "args": {"kind": "cafe"}},
    "Какие кафе есть на Пушкинской улице?": {
        "tool": "find_places", "args": {"kind": "cafe", "street": "Пушкин"}},
    "Какая аптека ближе всего к точке 47.2940, 39.8880?": {
        "tool": "nearest_place", "args": {"kind": "pharmacy"}},
    "Какая школа ближе всего к точке 47.2690, 39.7170?": {
        "tool": "nearest_place", "args": {"kind": "school"}},
    "Какие сети пунктов выдачи заказов представлены в городе?": {
        "tool": "top_brands", "args": {"kind": "outpost"}},
}


def correct_call(trace: dict) -> bool:
    """Первый вызов инструмента в траектории совпал с ожидаемым (имя + аргументы)."""
    exp = EXPECTED.get(trace["question"])
    if exp is None:
        return False
    calls = trace.get("tool_calls") or []
    if not calls:
        return False
    first = calls[0]
    if first.get("name") != exp["tool"]:
        return False
    got_args = first.get("args") or {}
    # Проверяем каждое ключевое поле аргументов.
    for key, want in exp["args"].items():
        got = got_args.get(key)
        # street сравниваем по вхождению (модель может написать «Пушкинская»).
        if key == "street":
            if not got or str(want).lower() not in str(got).lower():
                return False
        elif got != want:
            return False
    return True


def load_runs() -> list[dict]:
    """Все траектории из runs/*.jsonl (только штатные задачи из EXPECTED)."""
    traces = []
    for p in sorted(RUNS_DIR.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            # Отбрасываем сценарии-инъекции (failures.jsonl): их вопрос не из списка.
            if t.get("question") in EXPECTED:
                traces.append(t)
    return traces


def main() -> None:
    traces = load_runs()
    if not traces:
        print("Журналов нет — сначала запустите check.py.")
        return

    # Группируем по варианту описаний.
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for t in traces:
        by_variant[t.get("variant")].append(t)

    print("=" * 60)
    print("A/B ОПИСАНИЙ ИНСТРУМЕНТОВ")
    print("=" * 60)
    for variant in ("short", "full"):
        ts = by_variant.get(variant, [])
        n = len(ts)
        ok = sum(1 for t in ts if correct_call(t))
        p = ok / n if n else 0.0
        lo, hi = wilson_interval(p, n)
        # Средние токены на прогон в этом варианте (для вывода «окупились ли»).
        avg_in = sum(t.get("input_tokens", 0) for t in ts) / n if n else 0.0
        avg_out = sum(t.get("output_tokens", 0) for t in ts) / n if n else 0.0
        print(f"\nВариант: {variant}")
        print(f"  прогонов: {n}")
        print(f"  верный инструмент и аргументы: {ok}/{n} = {p:.3f}")
        print(f"  95% интервал Вильсона: [{lo}, {hi}]")
        print(f"  средние токены на прогон: {avg_in:.0f} вход / {avg_out:.0f} выход")

    # Сводный вывод про окупаемость токенов.
    short = by_variant.get("short", [])
    full = by_variant.get("full", [])
    if short and full:
        n_s, n_f = len(short), len(full)
        p_s = sum(correct_call(t) for t in short) / n_s
        p_f = sum(correct_call(t) for t in full) / n_f
        extra_tokens = (sum(t.get("input_tokens", 0) for t in full) / n_f
                        - sum(t.get("input_tokens", 0) for t in short) / n_s)
        print("\n" + "=" * 60)
        print("ВЫВОД")
        print("=" * 60)
        if p_f > p_s:
            print(f"«Полные» описания точнее: {p_f:.3f} против {p_s:.3f} "
                  f"(+{extra_tokens:.0f} входных токенов на прогон).")
            print("Лишние токены описаний окупились.")
        elif p_f < p_s:
            print(f"«Короткие» описания точнее: {p_s:.3f} против {p_f:.3f}. "
                  "Лишние токены полных описаний НЕ окупились.")
        else:
            print(f"Варианты неразличимы по точности ({p_s:.3f}). "
                  "Лишние токены полных описаний не дали выигрыша.")


if __name__ == "__main__":
    main()
