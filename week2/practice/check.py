#!/usr/bin/env python3
"""Прогон 10 задач через агента без фреймворка и подсчёт метрик.

Задание 1, пункт 2: «Прогнать десять задач по три раза: uv run python check.py --n 3.
В отчёт — pass@1 и pass^3, средние токены и стоимость задачи, число шагов».

Успех проверяется по типизированным данным финального JSON, а не по подстроке.
Стоимость измеряется токенами. Все попытки начинают с новой истории сообщений.

Запуск (из week2/practice):
    uv run python check.py --n 3              # вариант 'full'
    uv run python check.py --n 3 --variant short   # A/B-прогон
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import statistics
import sys
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from openai import OpenAI

# Чтобы `import sdk_tasks` и `from agent_loop import ...` работали независимо от
# рабочего каталога, добавляем каталог этого файла в начало sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sdk_tasks
from agent_loop import (RUNS_DIR, inside_practice, load_client, positive_int,
                        run_agent, save_run, valid_id)
from answer_schema import parse_answer


# ────────────────────────────────────────────────────────────────────────────
# 10 ЗАДАЧ С ЭТАЛОНАМИ
# ────────────────────────────────────────────────────────────────────────────
# Каждая задача: (вопрос, ожидаемое значение или обязательный элемент ответа).
# Эталоны пересчитаны по базе poi.sqlite (см. full_report.md, раздел «Эталоны»).
TASKS: list[tuple[str, str]] = [
    # 1. Счёт круглосуточных аптек: только count_places(kind=pharmacy, only_24_7=True).
    ("Сколько в Ростове круглосуточных аптек?", "19"),
    # 2. Счёт всех аптек.
    ("Сколько всего аптек в Ростове?", "330"),
    # 3. Счёт супермаркетов.
    ("Сколько супермаркетов в городе?", "352"),
    # 4. Счёт школ.
    ("Сколько школ в Ростове?", "107"),
    # 5. Счёт банков.
    ("Сколько банков в городе?", "204"),
    # 6. Счёт кафе.
    ("Сколько кафе в Ростове?", "261"),
    # 7. Список: кафе на Пушкинской — в ответе должно быть реальное название.
    ("Какие кафе есть на Пушкинской улице?", "Coffee Man"),
    # 8. Ближайшая аптека к точке: эталон — название изолированной аптеки (см. базу).
    ("Какая аптека ближе всего к точке 47.2940, 39.8880?", "Апрель"),
    # 9. Ближайшая школа к точке.
    ("Какая школа ближе всего к точке 47.2690, 39.7170?", "Школа №82"),
    # 10. Бренды пунктов выдачи: в топе должен быть Ozon.
    ("Какие сети пунктов выдачи заказов представлены в городе?", "Ozon"),
]


def is_solved(final_answer: dict | str | None, task_index: int) -> bool:
    """Grade outcome facts; the A/B tool-choice grader is deliberately separate."""
    try:
        answer = parse_answer(final_answer)
    except (ValueError, TypeError):
        return False
    if not 0 <= task_index < len(TASKS):
        raise ValueError("unknown_task")
    expected = TASKS[task_index][1]
    if task_index < 6:
        return answer["type"] == "count" and answer["count"] == int(expected)
    # Read-only connection: missing DB cannot create an empty replacement.
    with closing(sqlite3.connect(sdk_tasks.DB.resolve().as_uri() + "?mode=ro", uri=True)) as con:
        con.row_factory = sqlite3.Row
        if task_index == 6:
            if answer["type"] != "places":
                return False
            truth = {r["id"]: r["name"] for r in con.execute(
                "SELECT id, name FROM poi WHERE kind = ? AND street LIKE ?", ("cafe", "%Пушкин%"))}
            places = answer["places"]
            ids = [p["id"] for p in places]
            return (len(ids) == len(set(ids)) and any(p["name"] == expected for p in places)
                    and all(p["id"] in truth and p["name"] == truth[p["id"]] for p in places))
        if task_index in (7, 8):
            if answer["type"] != "nearest":
                return False
            lat, lon, kind = ((47.2940, 39.8880, "pharmacy") if task_index == 7
                              else (47.2690, 39.7170, "school"))
            places = list(con.execute("SELECT id, name, lat, lon FROM poi WHERE kind = ?", (kind,)))
            if not places:
                return False
            distance = lambda p: sdk_tasks.distance_m(lat, lon, p["lat"], p["lon"])
            best = min(places, key=distance)
            return (answer["place"]["id"] == best["id"]
                    and answer["place"]["name"] == best["name"]
                    and answer["distance_m"] == distance(best))
        if answer["type"] != "brands":
            return False
        truth = {r[0] for r in con.execute(
            "SELECT DISTINCT COALESCE(brand, 'без бренда') FROM poi WHERE kind = ?", ("outpost",))}
        brands = answer["brands"]
        return expected in brands and len(brands) == len(set(brands)) and set(brands) <= truth


# ────────────────────────────────────────────────────────────────────────────
# МЕТРИКИ pass@1 / pass^k
# ────────────────────────────────────────────────────────────────────────────
def pass_at_k(solved_matrix: list[list[bool]], k: int) -> float:
    """Доля задач, решённых хотя бы одной из первых k попыток.

    solved_matrix — список строк: по одной на задачу, по одной ячейке на попытку.
    pass@1 — доля задач, решённых с первой попытки; pass@3 — хотя бы одной из трёх.
    """
    _validate_matrix(solved_matrix, k)
    ok = sum(1 for attempts in solved_matrix if any(attempts[:k]))
    return ok / len(solved_matrix)


def _validate_matrix(matrix: list[list[bool]], k: int) -> None:
    if (type(k) is not int or k < 1 or not matrix
            or any(len(row) != len(matrix[0]) or len(row) < k
                   or any(type(x) is not bool for x in row) for row in matrix)):
        raise ValueError("incomplete_or_invalid_results_matrix")


def pass_power_k(solved_matrix: list[list[bool]], k: int) -> float:
    """Fraction of tasks for which ALL k attempts succeeded; not pass@1 ** k."""
    _validate_matrix(solved_matrix, k)
    return sum(all(row[:k]) for row in solved_matrix) / len(solved_matrix)


def wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% доверительный интервал Вильсона для доли p на выборке n.

    Формула (без поправки на непрерывность):
        центр  = (p + z²/2n) / (1 + z²/n)
        радиус = z/(1 + z²/n) * sqrt(p(1-p)/n + z²/4n²)
    Возвращает (нижняя, верхняя) границы. Используется в A/B (Фаза 4).
    """
    if type(n) is not int or n < 0 or not 0 <= p <= 1:
        raise ValueError("invalid_proportion")
    if n == 0:
        return 0.0, 1.0
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, center - radius), 4), round(min(1.0, center + radius), 4)


# ────────────────────────────────────────────────────────────────────────────
# ГЛАВНЫЙ ПРОГОН
# ────────────────────────────────────────────────────────────────────────────
def token_summary(traces: list[dict], n_tasks: int) -> dict:
    if not traces or n_tasks < 1:
        raise ValueError("empty_experiment")
    complete = all(t.get("usage_complete", True)
                   and type(t.get("input_tokens")) is int and t["input_tokens"] >= 0
                   and type(t.get("output_tokens")) is int and t["output_tokens"] >= 0
                   for t in traces)
    if not complete:
        return dict(avg_input=None, avg_output=None, avg_total=None, tokens_per_task=None,
                    unknown_usage=sum(not t.get("usage_complete", True)
                                      or t.get("input_tokens") is None
                                      or t.get("output_tokens") is None for t in traces))
    ins = sum(t["input_tokens"] for t in traces)
    outs = sum(t["output_tokens"] for t in traces)
    return dict(avg_input=ins / len(traces), avg_output=outs / len(traces),
                avg_total=(ins + outs) / len(traces), tokens_per_task=(ins + outs) / n_tasks,
                unknown_usage=0)


def run_suite(client: OpenAI, model: str, n: int, variant: str, max_steps: int,
              experiment_id: str) -> list[dict]:
    traces = []
    suite_fingerprint = hashlib.sha256(
        (json.dumps(TASKS, ensure_ascii=False) + hashlib.sha256(sdk_tasks.DB.read_bytes()).hexdigest()).encode()
    ).hexdigest()
    for index, (question, _) in enumerate(TASKS):
        for attempt in range(1, n + 1):
            trace = run_agent(client, model, question, variant, max_steps,
                              experiment_id=experiment_id, task_id=f"task-{index + 1:02d}",
                              attempt=attempt, planned_attempts=n)
            trace["suite_fingerprint"] = suite_fingerprint
            try:
                trace["solved"] = not trace["error"] and is_solved(trace["final_answer"], index)
                trace["evaluation_error"] = None
            except Exception:
                trace["solved"], trace["evaluation_error"] = False, "truth_unavailable"
            save_run(trace)
            traces.append(trace)
            # Status only: never dump question, model, response, config or traceback.
            print(f"{variant}: задача {index + 1}, попытка {attempt}: "
                  f"{'успех' if trace['solved'] else 'неуспех'}")
            if (trace["error"] or "").startswith("api_error") or trace["evaluation_error"]:
                raise RuntimeError("experiment_interrupted_partial_trace_saved")
    return traces


def main() -> None:
    ap = argparse.ArgumentParser(description="Прогон 10 задач и метрики")
    ap.add_argument("--n", type=positive_int, default=3, help="число прогонов на задачу (1–100)")
    ap.add_argument("--variant", choices=("short", "full", "both"), default="full",
                    help="вариант описаний инструментов (A/B)")
    ap.add_argument("--max-steps", type=positive_int, default=6, help="лимит шагов агента (1–32)")
    ap.add_argument("--experiment-id", type=valid_id, default=None)
    args = ap.parse_args()

    if args.n > 100 or args.max_steps > 32:
        ap.error("n должно быть 1–100, max-steps — 1–32")
    experiment_id = args.experiment_id or uuid4().hex
    variants = ("short", "full") if args.variant == "both" else (args.variant,)
    if inside_practice(RUNS_DIR / f"experiment_{experiment_id}.jsonl").exists():
        ap.error("идентификатор уже использован; выберите новый эксперимент")
    print(f"Эксперимент: {experiment_id}")
    try:
        client, model = load_client()
        with client:
            for variant in variants:
                traces = run_suite(client, model, args.n, variant, args.max_steps, experiment_id)
                solved = [[t["solved"] for t in traces if t["task_id"] == f"task-{i+1:02d}"]
                          for i in range(len(TASKS))]
                print(f"{variant}: pass@1={pass_at_k(solved, 1):.6f}; "
                      f"pass^{args.n}={pass_power_k(solved, args.n):.6f}; "
                      f"pass@{args.n}={pass_at_k(solved, args.n):.6f}")
                print("Расход токенов:", json.dumps(token_summary(traces, len(TASKS))))
                print(f"Среднее число шагов: {statistics.mean(t['steps'] for t in traces):.3f}")
    except Exception:
        print("Эксперимент не завершён. Доступные траектории сохранены; "
              "подробности ошибки скрыты. Итоговые метрики неполного эксперимента не выводятся.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
