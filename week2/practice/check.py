#!/usr/bin/env python3
"""Прогон 10 задач через агента без фреймворка и подсчёт метрик.

Задание 1, пункт 2: «Прогнать десять задач по три раза: uv run python check.py --n 3.
В отчёт — pass@1 и pass^3, средние токены и стоимость задачи, число шагов».

Что считается «решённой» задачей: финальный ответ агента содержит эталонное число
(или ключевую строку), которое пересчитывается по базе без участия модели. Задача
сформулирована так, чтобы правильный ответ зависел только от базы, а не от мнения.

Запуск (из week2/practice):
    uv run python check.py --n 3              # вариант 'full'
    uv run python check.py --n 3 --variant short   # A/B-прогон
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Чтобы `import sdk_tasks` и `from agent_loop import ...` работали независимо от
# рабочего каталога, добавляем каталог этого файла в начало sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sdk_tasks
from agent_loop import cost_rub, run_agent, save_run


# ────────────────────────────────────────────────────────────────────────────
# 10 ЗАДАЧ С ЭТАЛОНАМИ
# ────────────────────────────────────────────────────────────────────────────
# Каждая задача: (вопрос, эталон-строка). «Решена» = эталон встречается в ответе.
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


def is_solved(final_answer: str | None, expected: str) -> bool:
    """Задача решена, если эталонная строка есть в финальном ответе."""
    return bool(final_answer) and expected in final_answer


# ────────────────────────────────────────────────────────────────────────────
# МЕТРИКИ pass@1 / pass^k
# ────────────────────────────────────────────────────────────────────────────
def pass_at_k(solved_matrix: list[list[bool]], k: int) -> float:
    """Доля задач, решённых хотя бы одной из первых k попыток.

    solved_matrix — список строк: по одной на задачу, по одной ячейке на попытку.
    pass@1 — доля задач, решённых с первой попытки; pass^3 — хотя бы одной из трёх.
    """
    if not solved_matrix:
        return 0.0
    ok = sum(1 for attempts in solved_matrix if any(attempts[:k]))
    return round(ok / len(solved_matrix), 3)


def wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% доверительный интервал Вильсона для доли p на выборке n.

    Формула (без поправки на непрерывность):
        центр  = (p + z²/2n) / (1 + z²/n)
        радиус = z/(1 + z²/n) * sqrt(p(1-p)/n + z²/4n²)
    Возвращает (нижняя, верхняя) границы. Используется в A/B (Фаза 4).
    """
    if n == 0:
        return 0.0, 1.0
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, center - radius), 4), round(min(1.0, center + radius), 4)


# ────────────────────────────────────────────────────────────────────────────
# ГЛАВНЫЙ ПРОГОН
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Прогон 10 задач и метрики")
    ap.add_argument("--n", type=int, default=3, help="число прогонов на задачу")
    ap.add_argument("--variant", choices=("short", "full"), default="full",
                    help="вариант описаний инструментов (A/B)")
    ap.add_argument("--max-steps", type=int, default=6, help="лимит шагов агента")
    args = ap.parse_args()

    # Читаем .env (поднимаясь вверх до корня репозитория).
    for d in (Path.cwd(), *Path.cwd().parents):
        if (d / ".env").exists():
            load_dotenv(d / ".env")
            break

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ["OPENAI_BASE_URL"])
    model = os.environ["MODEL"]

    # Собираем матрицу результатов: rows = задачи, cols = попытки.
    solved: list[list[bool]] = []
    # Для средних токенов/стоимости/шагов копим сырые числа по всем прогонам.
    all_inputs: list[int] = []
    all_outputs: list[int] = []
    all_steps: list[int] = []

    print(f"Вариант описаний: {args.variant}, прогонов на задачу: {args.n}\n")
    for qi, (question, expected) in enumerate(TASKS, 1):
        attempts: list[bool] = []
        for attempt in range(1, args.n + 1):
            trace = run_agent(client, model, question, args.variant, args.max_steps)
            save_run(trace)
            ok = is_solved(trace["final_answer"], expected)
            attempts.append(ok)
            all_inputs.append(trace["input_tokens"])
            all_outputs.append(trace["output_tokens"])
            all_steps.append(trace["steps"])
            mark = "✓" if ok else "✗"
            print(f"  {qi:2d}.{attempt} {mark}  '{question}' → {trace['final_answer']!r}")
        solved.append(attempts)

    n_tasks = len(TASKS)
    total_runs = n_tasks * args.n
    avg_in = statistics.mean(all_inputs)
    avg_out = statistics.mean(all_outputs)
    avg_steps = statistics.mean(all_steps)
    # Средняя стоимость одной задачи = сумма стоимости всех прогонов / число задач.
    total_cost = sum(cost_rub(i, o) for i, o in zip(all_inputs, all_outputs))
    avg_cost_task = total_cost / n_tasks

    p1 = pass_at_k(solved, 1)
    pk = pass_at_k(solved, args.n)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТ")
    print("=" * 60)
    print(f"задач: {n_tasks}, прогонов на задачу: {args.n}")
    print(f"pass@1: {p1}")
    print(f"pass^{args.n}: {pk}")
    print(f"средние токены на прогон: {avg_in:.0f} вход / {avg_out:.0f} выход")
    print(f"средняя стоимость задачи: {avg_cost_task:.6f} ₽")
    print(f"среднее число шагов на прогон: {avg_steps:.1f}")
    print(f"всего прогонов: {total_runs}, общая стоимость: {total_cost:.6f} ₽")


if __name__ == "__main__":
    main()
