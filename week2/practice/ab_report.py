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
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

# Чтобы `from check import wilson_interval` работал из любого каталога, добавляем
# каталог этого файла в начало sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check import TASKS, token_summary, wilson_interval
from agent_loop import inside_practice, valid_id, validate_args

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
        "tool": "count_places", "args": {"kind": "pharmacy", "only_24_7": False}},
    "Сколько супермаркетов в городе?": {
        "tool": "count_places", "args": {"kind": "supermarket", "only_24_7": False}},
    "Сколько школ в Ростове?": {
        "tool": "count_places", "args": {"kind": "school", "only_24_7": False}},
    "Сколько банков в городе?": {
        "tool": "count_places", "args": {"kind": "bank", "only_24_7": False}},
    "Сколько кафе в Ростове?": {
        "tool": "count_places", "args": {"kind": "cafe", "only_24_7": False}},
    "Какие кафе есть на Пушкинской улице?": {
        "tool": "find_places", "args": {"kind": "cafe", "street": "Пушкин"}},
    "Какая аптека ближе всего к точке 47.2940, 39.8880?": {
        "tool": "nearest_place", "args": {"kind": "pharmacy", "lat": 47.2940, "lon": 39.8880}},
    "Какая школа ближе всего к точке 47.2690, 39.7170?": {
        "tool": "nearest_place", "args": {"kind": "school", "lat": 47.2690, "lon": 39.7170}},
    "Какие сети пунктов выдачи заказов представлены в городе?": {
        "tool": "top_brands", "args": {"kind": "outpost"}},
}


def correct_call(trace: dict) -> bool:
    """Первый вызов инструмента в траектории совпал с ожидаемым (имя + аргументы)."""
    exp = EXPECTED.get(trace.get("question"))
    if exp is None:
        return False
    calls = trace.get("tool_calls") or []
    if not calls:
        return False
    first = calls[0]
    if first.get("name") != exp["tool"]:
        return False
    got_args = first.get("args")
    try:
        validate_args(exp["tool"], got_args)
    except (ValueError, TypeError):
        return False
    got_args = {"only_24_7": False, **got_args}
    # Проверяем каждое ключевое поле аргументов.
    for key, want in exp["args"].items():
        got = got_args.get(key)
        # street сравниваем по вхождению (модель может написать «Пушкинская»).
        if key == "street":
            if not got or str(want).lower() not in str(got).lower():
                return False
        elif key in ("lat", "lon"):
            if type(got) not in (int, float) or not math.isclose(got, want, rel_tol=0, abs_tol=1e-5):
                return False
        elif type(got) is not type(want) or got != want:
            return False
    return True


def load_runs(experiment_id: str | None = None, *, legacy: bool = False) -> list[dict]:
    """Read one explicitly selected experiment; never pool unrelated JSONL files."""
    if legacy == (experiment_id is not None):
        raise ValueError("choose_experiment_or_legacy")
    path = inside_practice(RUNS_DIR / ("traces.jsonl" if legacy else
                                     f"experiment_{valid_id(experiment_id)}.jsonl"))
    traces = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            trace = json.loads(line)
        except ValueError:
            raise ValueError("invalid_journal_json") from None
        if not isinstance(trace, dict) or trace.get("question") not in EXPECTED:
            raise ValueError("unexpected_task_in_experiment")
        if not legacy and (trace.get("experiment_id") != experiment_id
                           or trace.get("format_version") != 2):
            raise ValueError("mixed_or_unsupported_experiment")
        if legacy and trace.get("format_version", 1) != 1:
            raise ValueError("not_a_legacy_experiment")
        traces.append(trace)
    return traces


def validate_experiment(traces: list[dict], *, legacy: bool = False) -> dict[str, list[dict]]:
    by_variant = defaultdict(list)
    for trace in traces:
        if trace.get("variant") not in ("short", "full"):
            raise ValueError("unknown_variant")
        by_variant[trace["variant"]].append(trace)
    if set(by_variant) != {"short", "full"}:
        raise ValueError("both_variants_required")
    if not legacy:
        for field in ("experiment_id", "format_version", "model_fingerprint", "protocol_fingerprint",
                      "implementation_fingerprint", "suite_fingerprint", "max_steps", "planned_attempts"):
            values = {t.get(field) for t in traces}
            if len(values) != 1 or None in values:
                raise ValueError("incompatible_experiment_metadata")
        ids = [t.get("trace_id") for t in traces]
        if None in ids or len(set(ids)) != len(ids):
            raise ValueError("duplicate_or_missing_trace_id")
    for variant, ts in by_variant.items():
        n = 3 if legacy else ts[0]["planned_attempts"]
        if type(n) is not int or n < 1:
            raise ValueError("invalid_attempt_count")
        groups = defaultdict(list)
        for trace in ts:
            groups[trace.get("question")].append(trace)
        if set(groups) != set(EXPECTED) or any(len(g) != n for g in groups.values()):
            raise ValueError("incomplete_or_duplicate_experiment")
        if not legacy:
            fingerprints = {t.get("descriptions_fingerprint") for t in ts}
            if len(fingerprints) != 1 or None in fingerprints:
                raise ValueError("mixed_descriptions")
            for index, (question, _) in enumerate(TASKS):
                rows = groups[question]
                if (any(t.get("task_id") != f"task-{index + 1:02d}" for t in rows)
                        or any(type(t.get("attempt")) is not int for t in rows)
                        or {t["attempt"] for t in rows} != set(range(1, n + 1))):
                    raise ValueError("invalid_task_or_attempt_identifiers")
    return dict(by_variant)


def summarize(traces: list[dict], *, legacy: bool = False) -> dict:
    grouped = validate_experiment(traces, legacy=legacy)
    result = {}
    for variant, ts in grouped.items():
        ok = sum(correct_call(t) for t in ts)
        p = ok / len(ts)
        tokens = token_summary(ts, len(TASKS))
        result[variant] = dict(n=len(ts), correct=ok, accuracy=p,
                               wilson_95=wilson_interval(p, len(ts)), **tokens,
                               tokens_per_correct_choice=(tokens["avg_total"] / p
                                   if p and tokens["avg_total"] is not None else None))
    short, full = result["short"], result["full"]
    extra = (full["avg_total"] - short["avg_total"]
             if full["avg_total"] is not None and short["avg_total"] is not None else None)
    delta = full["accuracy"] - short["accuracy"]
    result["comparison"] = dict(delta_accuracy=delta, extra_tokens_per_attempt=extra,
        extra_tokens_per_extra_correct_choice=extra / delta if extra is not None and delta > 0 else None)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B одного эксперимента; без API-вызовов")
    choice = ap.add_mutually_exclusive_group(required=True)
    choice.add_argument("--experiment-id", type=valid_id)
    choice.add_argument("--legacy", action="store_true", help="исторический runs/traces.jsonl")
    args = ap.parse_args()
    try:
        report = summarize(load_runs(args.experiment_id, legacy=args.legacy), legacy=args.legacy)
    except Exception:
        print("Журнал отсутствует, повреждён, неполон или содержит несовместимые эксперименты.")
        raise SystemExit(1) from None
    print("ИСТОРИЧЕСКИЕ ДАННЫЕ: старый свободный ответ, другая конфигурация; "
          "не оценка новой модели." if args.legacy else "A/B выбранного эксперимента")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Интервал Вильсона — описательная оценка объединённых попыток; "
          "30 попыток на 10 задачах не доказывают статистическую значимость разницы.")


if __name__ == "__main__":
    main()
