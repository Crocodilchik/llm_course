#!/usr/bin/env python3
"""Генератор неудачных траекторий для разбора ошибок (задание 1, пункт 4).

Агент на 10 штатных задачах точен (pass@1=1.0), поэтому честных провалов в журнале
нет. Чтобы показать классификацию ошибок, этот скрипт намеренно прогоняет три
сценария, каждый из которых даёт траекторию своего класса:

  1. ОШИБКА СРЕДЫ — инструмент бросает исключение. Проверяем, что цикл не падает,
     а превращает исключение в наблюдение «ОШИБКА: ...» и даёт модели шанс.
  2. ЗАЦИКЛИВАНИЕ — упрямый инструмент, который вечно просит уточнить. Проверяем,
     что лимит шагов останавливает прогон (stopped_by_limit=True).
  3. ЛОЖНОЕ «ГОТОВО» — вопрос о данных, которых в базе нет; смотрим, выдумывает ли
     модель ответ, не позвав инструмент (или честно ли говорит, что данных нет).

Все три траектории пишутся в runs/ (файл failures.jsonl), чтобы их можно было
приложить к отчёту. Это вспомогательный скрипт; в минимальную сдачу он не входит.

Запуск (из week2/practice):
    uv run python failures.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Чтобы импорты agent_loop/sdk_tasks работали из любого каталога.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent_loop
import sdk_tasks

# Копии штатных описаний: сценарий 2 временно подменяет их и должен вернуть как было.
DESCRIPTIONS_SHORT = dict(agent_loop.DESCRIPTIONS["short"])
DESCRIPTIONS_FULL = dict(agent_loop.DESCRIPTIONS["full"])


def load_env() -> tuple[OpenAI, str]:
    """Возвращает клиент и имя модели, прочитав .env (подъёмом вверх)."""
    for d in (Path.cwd(), *Path.cwd().parents):
        if (d / ".env").exists():
            load_dotenv(d / ".env")
            break
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ["OPENAI_BASE_URL"])
    return client, os.environ["MODEL"]


def save(trace: dict) -> None:
    """Дописывает траекторию в runs/failures.jsonl."""
    out = agent_loop.RUNS_DIR / "failures.jsonl"
    agent_loop.RUNS_DIR.mkdir(exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")


def scenario_env_error(client: OpenAI, model: str) -> dict:
    """Сценарий 1: инструмент бросает исключение -> ошибка как наблюдение.

    Временно добавляем в agent_loop «сломанный» инструмент, который делит на ноль,
    и задаём вопрос, который его спровоцирует. Цикл должен поймать исключение,
    отдать модели «ОШИБКА: ZeroDivisionError ...», и модель должна как-то выйти
    из ситуации, не уронив прогон.
    """
    agent_loop.EXECUTORS["broken_divide"] = lambda x=0: 1 / x          # всегда падает
    agent_loop.SCHEMAS["broken_divide"] = {"type": "object", "properties": {}}
    agent_loop.DESCRIPTIONS["short"]["broken_divide"] = "Делит на число."
    agent_loop.DESCRIPTIONS["full"]["broken_divide"] = "Делит на число."
    # make_tools итерирует по EXECUTORS, поэтому новый инструмент подхватится.
    trace = agent_loop.run_agent(client, model,
                                 "Позови инструмент broken_divide.", "full", 6)
    # Убираем временный инструмент, чтобы не мусорить в штатных прогонах.
    del agent_loop.EXECUTORS["broken_divide"]
    del agent_loop.SCHEMAS["broken_divide"]
    del agent_loop.DESCRIPTIONS["short"]["broken_divide"]
    del agent_loop.DESCRIPTIONS["full"]["broken_divide"]
    trace["scenario"] = "env_error"
    return trace


def scenario_infinite_loop(client: OpenAI, model: str) -> dict:
    """Сценарий 2: упрямый инструмент -> зацикливание -> остановка лимитом.

    sdk_tasks.stubborn_tool на любой запрос просит уточнить фильтр, поэтому модель
    зовёт его снова и снова. Чтобы модель действительно зациклилась (а не ответила
    текстом), временно подменяем системную инструкцию на жёсткую — «всегда зови
    инструмент». Лимит шагов должен оборвать цикл.
    """
    # Запоминаем штатные инструменты и оставляем ТОЛЬКО упрямый: тогда модели некуда
    # переключиться, и она будет звать его до упора — это и есть зацикливание.
    saved_executors = agent_loop.EXECUTORS
    saved_schemas = agent_loop.SCHEMAS
    saved_system = agent_loop.SYSTEM
    agent_loop.SYSTEM = (
        "Ты обязан вызывать инструмент поиска на каждом шаге. Тебе запрещено "
        "отвечать обычным текстом. Всегда уточняй фильтр и вызывай инструмент снова."
    )
    agent_loop.EXECUTORS = {"stubborn": sdk_tasks.stubborn_tool}
    agent_loop.SCHEMAS = {"stubborn": {"type": "object", "properties": {
        "kind": {"type": "string", "description": "вид места"}}}}
    agent_loop.DESCRIPTIONS["short"] = {"stubborn": "Ищет места."}
    agent_loop.DESCRIPTIONS["full"] = {"stubborn": "Ищет места."}
    try:
        trace = agent_loop.run_agent(client, model, "Найди места в городе", "full", 6)
    finally:
        # Возвращаем штатное состояние, чтобы не мусорить в других прогонах.
        agent_loop.SYSTEM = saved_system
        agent_loop.EXECUTORS = saved_executors
        agent_loop.SCHEMAS = saved_schemas
        agent_loop.DESCRIPTIONS["short"] = DESCRIPTIONS_SHORT
        agent_loop.DESCRIPTIONS["full"] = DESCRIPTIONS_FULL
    trace["scenario"] = "infinite_loop"
    return trace


def scenario_false_ready(client: OpenAI, model: str) -> dict:
    """Сценарий 3: вопрос без опоры на инструмент -> ложное «готово».

    Вопрос «где лучше открыть кофейню» не решается ни одним из пяти инструментов
    (они дают факты, а не рекомендации). Смотрим, выдумает ли модель ответ по памяти
    без вызова инструмента — это и есть ложное «готово»: уверенный ответ, который
    ничем не сверен с базой.
    """
    trace = agent_loop.run_agent(
        client, model,
        "Где в Ростове лучше всего открыть новую кофейню и почему?", "full", 6)
    trace["scenario"] = "false_ready"
    return trace


def main() -> None:
    client, model = load_env()
    for name, fn in (("env_error", scenario_env_error),
                     ("infinite_loop", scenario_infinite_loop),
                     ("false_ready", scenario_false_ready)):
        trace = fn(client, model)
        save(trace)
        print(f"\n=== {name} ===")
        print("  шагов:", trace["steps"])
        print("  финальный ответ:", repr(trace["final_answer"]))
        print("  остановлен лимитом:", trace["stopped_by_limit"])
        print("  error:", trace["error"])
        print("  вызовы:", [(c["name"], c.get("error")) for c in trace["tool_calls"]])


if __name__ == "__main__":
    main()
