#!/usr/bin/env python3
"""Агент без фреймворка: модель в цикле с инструментами над базой Ростова.

Здесь нет openai-agents — только клиент openai (chat.completions) и собственный
цикл. Это то, что задание 1 требует собрать руками: инструменты, лимит шагов,
ошибки как наблюдения, журнал в JSONL и счётчики токенов.

Файл делится на четыре части:
1. ЦЕНА — тариф polza.ai (рублей за миллион токенов), по нему считаем стоимость.
2. ИНСТРУМЕНТЫ — пять функций над базой + их схемы и два варианта описаний
   (short = одна фраза, full = что/когда/когда-НЕ/единицы/пример) для A/B.
3. ЦИКЛ — run_agent(): модель -> tool_calls -> исполнение -> наблюдение -> повтор.
4. ЖУРНАЛ — save_run(): каждая траектория пишется одной строкой JSON в runs/.

Запуск из week2/practice:  uv run python agent_loop.py "Сколько аптек в Ростове?"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from pathlib import Path

from openai import OpenAI

# Чтобы `import sdk_tasks` работал независимо от того, откуда запущен скрипт,
# добавляем каталог этого файла (week2/practice) в начало sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sdk_tasks  # готовые функции над базой (даны курсом)


# ────────────────────────────────────────────────────────────────────────────
# 1. ЦЕНА ТОКЕНОВ
# ────────────────────────────────────────────────────────────────────────────
# Тариф polza.ai для выбранной модели: рублей за МИЛЛИОН токенов.
PRICE_IN = 13.112    # входные токены (запрос)
PRICE_OUT = 39.335   # выходные токены (ответ)


def cost_rub(in_tokens: int, out_tokens: int) -> float:
    """Стоимость прогона в рублях по количеству входных и выходных токенов."""
    return round(in_tokens / 1e6 * PRICE_IN + out_tokens / 1e6 * PRICE_OUT, 6)


# ────────────────────────────────────────────────────────────────────────────
# 2. ИНСТРУМЕНТЫ
# ────────────────────────────────────────────────────────────────────────────
# Исполнители — обёртки над sdk_tasks. Каждая возвращает СТРОКУ: это и есть
# «текст результата», который модель увидит как наблюдение (observation).
def _count_places(kind: str, only_24_7: bool = False) -> str:
    """Сколько мест вида kind в городе; при only_24_7 — только круглосуточные."""
    return sdk_tasks.count_places(kind, only_24_7)


def _find_places(kind: str, street: str | None = None, limit: int = 10) -> str:
    """Список мест вида kind (при street — на одной улице)."""
    return sdk_tasks.find_places(kind, street, limit)


def _nearest_place(lat: float, lon: float, kind: str) -> str:
    """Ближайшее к точке (lat, lon) место вида kind с расстоянием в метрах."""
    found = sdk_tasks.nearest(lat, lon, kind, 1)
    if not found:
        return f"мест вида {kind} в базе нет"
    b = found[0]
    return f"{b['name'] or 'без названия'} — {b['distance_m']} м (id={b['id']})"


def _place_details(place_id: int) -> str:
    """Карточка места по id: название, часы, телефон, адрес."""
    card = sdk_tasks.details(place_id)
    if not card:
        return "места с таким id в базе нет"
    return f"{card['name']} · часы {card['opening_hours']} · тел {card['phone']}"


def _top_brands(kind: str, limit: int = 5) -> str:
    """Самые частые бренды/сети среди мест вида kind."""
    return sdk_tasks.top_brands(kind, limit)


# Исполнители по имени инструмента.
EXECUTORS = {
    "count_places": _count_places,
    "find_places": _find_places,
    "nearest_place": _nearest_place,
    "place_details": _place_details,
    "top_brands": _top_brands,
}


# Схемы аргументов (JSON Schema) — одинаковы для обоих вариантов описаний.
SCHEMAS = {
    "count_places": {
        "type": "object",
        "required": ["kind"],
        "properties": {
            "kind": {"type": "string",
                     "description": "вид места латиницей: pharmacy, cafe, supermarket, ..."},
            "only_24_7": {"type": "boolean",
                          "description": "считать только круглосуточные"},
        },
    },
    "find_places": {
        "type": "object",
        "required": ["kind"],
        "properties": {
            "kind": {"type": "string",
                     "description": "вид места латиницей: pharmacy, cafe, supermarket, ..."},
            "street": {"type": "string",
                       "description": "часть названия улицы; без неё ищется по всему городу"},
            "limit": {"type": "integer", "description": "сколько строк вернуть"},
        },
    },
    "nearest_place": {
        "type": "object",
        "required": ["lat", "lon", "kind"],
        "properties": {
            "lat": {"type": "number", "description": "широта точки"},
            "lon": {"type": "number", "description": "долгота точки"},
            "kind": {"type": "string",
                     "description": "вид места латиницей: pharmacy, cafe, outpost, ..."},
        },
    },
    "place_details": {
        "type": "object",
        "required": ["place_id"],
        "properties": {
            "place_id": {"type": "integer", "description": "идентификатор места"},
        },
    },
    "top_brands": {
        "type": "object",
        "required": ["kind"],
        "properties": {
            "kind": {"type": "string",
                     "description": "вид места латиницей: pharmacy, outpost, ..."},
            "limit": {"type": "integer", "description": "сколько брендов вернуть"},
        },
    },
}


# Описания инструментов: два варианта для A/B.
# short — одна фраза; full — что делает / когда звать / когда НЕ звать /
# единицы / пример. Меняется только description, схемы и код одинаковы.
DESCRIPTIONS = {
    "short": {
        "count_places": "Сколько в городе мест заданного вида.",
        "find_places": "Список мест заданного вида.",
        "nearest_place": "Ближайшее к точке место заданного вида.",
        "place_details": "Карточка места по id.",
        "top_brands": "Самые частые бренды вида.",
    },
    "full": {
        "count_places": (
            "Возвращает количество мест заданного вида в базе города — одно число. "
            "Зови на вопросы «сколько», «какое количество». "
            "НЕ зови, когда нужен перечень заведений — для этого есть find_places. "
            "Единицы: kind — вид места латиницей (pharmacy, cafe, supermarket). "
            "Пример: «Сколько аптек в Ростове?» → kind='pharmacy'."
        ),
        "find_places": (
            "Возвращает список мест заданного вида, при желании — только на одной улице. "
            "Зови, когда просят перечислить заведения: какие аптеки/кафе/магазины есть. "
            "НЕ зови для вопросов «сколько» (это count_places) и «что ближе» "
            "(это nearest_place). "
            "Единицы: kind — вид места латиницей, street — часть названия улицы. "
            "Пример: «Какие кафе на Пушкинской?» → kind='cafe', street='Пушкин'."
        ),
        "nearest_place": (
            "Возвращает ближайшее к заданной точке место указанного вида и расстояние "
            "до него в метрах. "
            "Зови на «что ближе», «сколько метров до», когда даны координаты. "
            "НЕ зови для перечня мест (find_places) и для «сколько» (count_places). "
            "Единицы: lat/lon — координаты точки, kind — вид места латиницей. "
            "Пример: «Какая аптека ближе к 47.2357, 39.7015?» → lat=47.2357, "
            "lon=39.7015, kind='pharmacy'."
        ),
        "place_details": (
            "Возвращает карточку места по его идентификатору: название, часы работы, "
            "телефон, адрес. "
            "Зови, когда уже известен id места и нужны его детали. "
            "НЕ зови без id — его даёт find_places или nearest_place. "
            "Единицы: place_id — целое число из ответа другого инструмента."
        ),
        "top_brands": (
            "Возвращает самые частые бренды/сети среди мест заданного вида. "
            "Зови, когда спрашивают, какие сети или бренды представлены в городе. "
            "НЕ зови для списка конкретных заведений (find_places) или их числа "
            "(count_places). "
            "Единицы: kind — вид места латиницей. "
            "Пример: «Какие сети пунктов выдачи?» → kind='outpost'."
        ),
    },
}


def make_tools(variant: str) -> list[dict]:
    """Собирает список инструментов в формате OpenAI tools для заданного варианта.

    variant — 'short' или 'full'; от него зависит только description.
    """
    descs = DESCRIPTIONS[variant]
    return [
        {"type": "function", "function": {
            "name": name, "description": descs[name], "parameters": SCHEMAS[name],
        }}
        for name in EXECUTORS
    ]


# ────────────────────────────────────────────────────────────────────────────
# 3. ЦИКЛ
# ────────────────────────────────────────────────────────────────────────────
SYSTEM = (
    "Ты — городская справочная по Ростову-на-Дону. Отвечай только по данным базы "
    "мест города, вызывая инструменты. Не выдумывай числа и названия: бери их из "
    "ответов инструментов. Отвечай по-русски."
)


def run_agent(client: OpenAI, model: str, question: str, variant: str,
              max_steps: int = 6) -> dict:
    """Один прогон агента. Возвращает словарь-траекторию.

    Логика цикла: отправляем сообщения модели; если она вернула tool_calls —
    исполняем каждый инструмент и добавляем результат как tool-сообщение; если
    финальный текст — останавливаемся. Исключение инструмента не роняет цикл,
    а превращается в наблюдение «ОШИБКА: ...», чтобы модель могла исправиться.
    """
    tools = make_tools(variant)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    trace = {
        "question": question,
        "variant": variant,
        "max_steps": max_steps,
        "steps": 0,                 # сколько итераций (вызовов модели) сделано
        "tool_calls": [],           # журнал вызовов инструментов
        "final_answer": None,
        "input_tokens": 0,          # сумма входных токенов за прогон
        "output_tokens": 0,         # сумма выходных токенов
        "error": None,
        "stopped_by_limit": False,  # упёрлись в лимит шагов
    }

    for step in range(1, max_steps + 1):
        trace["steps"] = step
        # Один вызов модели. tools передаём всегда — иначе модель их не увидит.
        resp = client.chat.completions.create(model=model, messages=messages, tools=tools)
        # Копим токены из usage (если провайдер их вернул).
        if getattr(resp, "usage", None):
            trace["input_tokens"] += resp.usage.prompt_tokens or 0
            trace["output_tokens"] += resp.usage.completion_tokens or 0

        msg = resp.choices[0].message

        # Модель вернула финальный текст — прогон завершён.
        if not msg.tool_calls:
            trace["final_answer"] = msg.content
            return trace

        # Модель попросила вызвать инструменты. Добавляем её ответ в историю.
        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})

        # Исполняем каждый запрошенный инструмент.
        for tc in msg.tool_calls:
            name = tc.function.name
            # Аргументы приходят строкой JSON — парсим, падая к пустому словарю.
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            # Ошибка инструмента не роняет цикл, а становится наблюдением.
            try:
                result = EXECUTORS[name](**args)
                err = None
            except Exception as exc:  # noqa: BLE001 — любое исключение инструмента
                result = f"ОШИБКА: {type(exc).__name__}: {exc}"
                err = result
            trace["tool_calls"].append({"name": name, "args": args, "result": result,
                                        "error": err})
            # Результат уходит модели как tool-сообщение.
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Цикл исчерпан: лимит шагов сработал, финального ответа нет.
    trace["stopped_by_limit"] = True
    trace["error"] = f"достигнут лимит шагов ({max_steps}) без финального ответа"
    return trace


# ────────────────────────────────────────────────────────────────────────────
# 4. ЖУРНАЛ
# ────────────────────────────────────────────────────────────────────────────
RUNS_DIR = Path(__file__).with_name("runs")


def save_run(trace: dict) -> Path:
    """Дописывает траекторию одной строкой JSON в runs/<timestamp>.jsonl."""
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"run_{int(time.time())}.jsonl"
    # Ключ-порядок не гарантирован — сериализуем явно, ensure_ascii=False для кириллицы.
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")
    return path


def main() -> None:
    """CLI-режим: один вопрос -> один прогон -> печать ответа и стоимости."""
    ap = argparse.ArgumentParser(description="Агент без фреймворка над базой Ростова")
    ap.add_argument("question", help="вопрос агенту")
    ap.add_argument("--variant", choices=("short", "full"), default="full",
                    help="вариант описаний инструментов (A/B)")
    ap.add_argument("--max-steps", type=int, default=6, help="лимит шагов")
    args = ap.parse_args()

    # Читаем .env (поднимаясь вверх до корня репозитория).
    from dotenv import load_dotenv
    for d in (Path.cwd(), *Path.cwd().parents):
        if (d / ".env").exists():
            load_dotenv(d / ".env")
            break

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ["OPENAI_BASE_URL"])
    trace = run_agent(client, os.environ["MODEL"], args.question,
                      args.variant, args.max_steps)
    print("Ответ:", trace["final_answer"])
    print(f"Шагов: {trace['steps']}, токенов: {trace['input_tokens']} вх / "
          f"{trace['output_tokens']} вых, "
          f"стоимость: {cost_rub(trace['input_tokens'], trace['output_tokens'])} ₽")
    save_run(trace)


if __name__ == "__main__":
    main()
