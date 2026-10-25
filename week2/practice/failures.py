#!/usr/bin/env python3
"""Поиск трёх проверяемых неудачных траекторий (задание 1, пункт 4).

По умолчанию: до 18 попыток штатного агента на шести разрешимых задачах;
при недоборе — до трёх явно помеченных отключений хранилища в памяти.
Независимый SQLite-эталон проверяет исход; все попытки сохраняются.
Три прежних сценария ниже доступны только с --legacy-scenarios:

Исторические штатные задачи оценивались другим проверяющим кодом и моделью.
Чтобы показать классификацию ошибок, этот скрипт намеренно прогоняет три
сценария, каждый из которых даёт траекторию своего класса:

  1. ОШИБКА СРЕДЫ — инструмент бросает исключение. Проверяем, что цикл не падает,
     а превращает исключение в наблюдение «ОШИБКА: ...» и даёт модели шанс.
  2. ЗАЦИКЛИВАНИЕ — упрямый инструмент, который вечно просит уточнить. Проверяем,
     что лимит шагов останавливает прогон (stopped_by_limit=True).
  3. ЛОЖНОЕ «ГОТОВО» — вопрос о данных, которых в базе нет; смотрим, выдумывает ли
     модель ответ, не позвав инструмент (или честно ли говорит, что данных нет).

Новые траектории пишутся в отдельный experiment_<id>.jsonl. Исторический
failures.jsonl не изменяется. Сценарий не гарантирует заранее выбранный исход:
в отчёте классифицируется фактически наблюдавшееся поведение.

Запуск (из week2/practice):
    uv run python failures.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from contextlib import closing, contextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from openai import OpenAI

# Чтобы импорты agent_loop/sdk_tasks работали из любого каталога.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent_loop
import sdk_tasks
from answer_schema import parse_answer

# Копии штатных описаний: сценарий 2 временно подменяет их и должен вернуть как было.
DESCRIPTIONS_SHORT = dict(agent_loop.DESCRIPTIONS["short"])
DESCRIPTIONS_FULL = dict(agent_loop.DESCRIPTIONS["full"])


def load_env() -> tuple[OpenAI, str]:
    return agent_loop.load_client()


def save(trace: dict) -> None:
    agent_loop.save_run(trace)


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
    try:
        trace = agent_loop.run_agent(client, model,
                                     "Позови инструмент broken_divide.", "full", 6)
    finally:
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


def legacy_main() -> None:
    experiment_id = "failures-" + uuid4().hex
    try:
        client, model = load_env()
        with client:
            for name, fn in (("env_error", scenario_env_error),
                             ("infinite_loop", scenario_infinite_loop),
                             ("false_ready", scenario_false_ready)):
                trace = fn(client, model)
                trace.update(experiment_id=experiment_id, task_id=name, attempt=1, planned_attempts=1)
                save(trace)
                print(f"{name}: шагов={trace['steps']}, "
                      f"лимит={trace['stopped_by_limit']}, ошибка={trace['error']}")
        print("Диагностический эксперимент:", experiment_id)
    except Exception:
        print("Диагностика прервана; подробности ошибки скрыты.")
        raise SystemExit(1) from None


# Separate diagnostic tasks: the ten benchmark tasks and A/B are unchanged.
# Only facts representable by the existing answer contract are requested.
SEARCH_TASKS = (
    dict(id="card-pharmacy", kind="pharmacy", lat=47.2940, lon=39.8880, answer="details",
         question="Найди ближайшую аптеку к широте 47.2940, долготе 39.8880. Верни её карточку: id, название, адрес и часы работы из базы; отсутствующие поля оставь null."),
    dict(id="card-school", kind="school", lat=47.2690, lon=39.7170, answer="details",
         question="Нужны адрес и часы работы ближайшей школы к широте 47.2690, долготе 39.7170. Найди школу, затем её карточку; верни details с id и названием. Отсутствующие поля — null."),
    dict(id="card-bank-reversed", kind="bank", lat=47.2357, lon=39.7015, answer="details",
         question="Точка задана в порядке долгота, широта: 39.7015, 47.2357 градуса. Найди ближайший банк, затем верни его карточку details с id, названием, адресом и часами из базы. Отсутствующие поля — null."),
    dict(id="nearest-reversed", kind="pharmacy", lat=47.2940, lon=39.8880, answer="nearest",
         question="Координаты записаны как долгота 39.8880, широта 47.2940 градуса. Какая аптека ближайшая? Верни id, название и расстояние в метрах по инструменту."),
    dict(id="nearest-minutes", kind="school", lat=47.2690, lon=39.7170, answer="nearest",
         question="Какая школа ближайшая к точке 47 градусов 16.14 минуты северной широты, 39 градусов 43.02 минуты восточной долготы? Переведи в десятичные градусы и верни id, название, расстояние в метрах по инструменту."),
    dict(id="all-outpost-brands", kind="outpost", answer="brands",
         question="Перечисли все различные сети пунктов выдачи в базе, а не только первые пять. Включи категорию «без бренда», если она есть. Верни brands без повторов."),
)


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def expected_answer(task: dict) -> dict:
    """Independent read-only oracle; never call a possibly injected executor."""
    with closing(sqlite3.connect(sdk_tasks.DB.resolve().as_uri() + "?mode=ro", uri=True)) as con:
        con.row_factory = sqlite3.Row
        if task["answer"] == "count":
            n = con.execute("SELECT COUNT(*) FROM poi WHERE kind=?", (task["kind"],)).fetchone()[0]
            return dict(type="count", count=n)
        if task["answer"] == "brands":
            brands = [r[0] for r in con.execute(
                "SELECT DISTINCT COALESCE(brand, 'без бренда') FROM poi WHERE kind=?",
                (task["kind"],))]
            if not 1 <= len(brands) <= 50:
                raise ValueError("diagnostic_answer_not_representable")
            return dict(type="brands", brands=sorted(brands))
        candidates = list(con.execute(
            "SELECT id, name, lat, lon, street, housenumber, opening_hours FROM poi WHERE kind=?",
            (task["kind"],)))
        if not candidates:
            raise ValueError("diagnostic_truth_missing")
        # Same documented metric as sdk_tasks.distance_m, independently computed.
        def distance(row):
            return int(math.hypot((task["lat"] - row["lat"]) * 111_320,
                (task["lon"] - row["lon"]) * 111_320 * math.cos(math.radians(47.23))))
        nearest = min(candidates, key=distance)
        place = dict(id=nearest["id"], name=nearest["name"])
        if task["answer"] == "nearest":
            return dict(type="nearest", place=place, distance_m=distance(nearest))
        address = " ".join(str(nearest[k]) for k in ("street", "housenumber") if nearest[k])
        return dict(type="details", place=place, address=address or None,
                    opening_hours=nearest["opening_hours"] or None)


def correct_answer(value, expected: dict) -> bool:
    try:
        answer = parse_answer(value)
    except (TypeError, ValueError):
        return False
    if answer["type"] != expected["type"]:
        return False
    if answer["type"] == "brands":
        return len(answer["brands"]) == len(set(answer["brands"])) and set(answer["brands"]) == set(expected["brands"])
    # The tool renders an absent name as «без названия»; both representations
    # express the same missing value. Do not penalize this presentation choice.
    if "place" in answer and expected["place"]["name"] is None:
        if answer["place"]["name"] == "без названия":
            answer["place"]["name"] = None
    return answer == expected


@contextmanager
def unavailable_storage(enabled: bool):
    """Simulate a backend outage in memory; leave the DB and protocol intact."""
    saved = agent_loop.EXECUTORS
    if enabled:
        def unavailable(**kwargs):
            raise OSError("diagnostic_storage_unavailable")
        agent_loop.EXECUTORS = {name: unavailable for name in saved}
    try:
        yield
    finally:
        agent_loop.EXECUTORS = saved


def assess(trace: dict, expected: dict, injected: bool) -> dict:
    """Separate task failure, safe handling, and unassessed API failures."""
    facts_correct = correct_answer(trace.get("final_answer"), expected)
    calls = trace.get("tool_calls", [])
    errors = [c.get("error") for c in calls if c.get("error")]
    successful = any(not c.get("error") for c in calls)
    if (trace.get("error") or "").startswith("api_error"):
        return dict(solved=None, outcome="unassessed", failure_class=None,
                    failure_reason="api_failure", facts_correct=facts_correct)
    # All questions require verification against the DB. A lucky guess during
    # a complete backend outage is not a verified answer to that goal.
    solved = facts_correct and not trace.get("error") and successful
    if solved:
        category, reason = None, None
    elif injected and any(e.startswith("tool_error:") for e in errors):
        category, reason = "environment_error", "controlled_storage_outage"
    elif trace.get("stopped_by_limit"):
        category, reason = "looping", "step_limit"
    elif any(e.startswith("tool_error:") or e == "result_too_large" for e in errors):
        category, reason = "environment_error", "tool_failure_without_recovery"
    elif "invalid_arguments" in errors:
        category, reason = "wrong_arguments", "invalid_arguments_without_recovery"
    elif (trace.get("final_answer") or {}).get("type") == "unavailable":
        category, reason = "incomplete_task", "answerable_task_refused"
    else:
        category, reason = "false_done", "incorrect_or_unverified_final_answer"
    return dict(solved=solved, outcome="solved" if solved else "failed",
                failure_class=category, failure_reason=reason, facts_correct=facts_correct)


def search(client: OpenAI, model: str, experiment_id: str) -> list[dict]:
    """At most 18 natural attempts, then at most 3 explicit injected cases."""
    experiment_id = agent_loop.valid_id(experiment_id)
    path = agent_loop.inside_practice(agent_loop.RUNS_DIR / f"experiment_{experiment_id}.jsonl")
    if path.exists():
        raise ValueError("experiment_already_exists")
    fallback_count = dict(id="outage-count", kind="bank", answer="count",
        question="Проверь по базе, сколько всего банков в городе. Нужен точный count из инструментов, без догадок.")
    fallback = (fallback_count, {**SEARCH_TASKS[3], "id": "outage-nearest"},
                {**SEARCH_TASKS[0], "id": "outage-details"})
    # Preflight all oracles before spending any model requests.
    tasks = (*SEARCH_TASKS, *fallback)
    truth = {t["id"]: expected_answer(t) for t in tasks}
    metadata = dict(diagnostic_version=1, diagnostic_target=3,
        diagnostic_policy="natural_then_controlled_outage",
        diagnostic_implementation_fingerprint=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        diagnostic_suite_fingerprint=fingerprint([tasks, truth,
            hashlib.sha256(sdk_tasks.DB.read_bytes()).hexdigest(), 3, 6]))
    traces = []
    for injected, cases, attempts in ((False, SEARCH_TASKS, 3), (True, fallback, 1)):
        for task in cases:
            for attempt in range(1, attempts + 1):
                with unavailable_storage(injected):
                    trace = agent_loop.run_agent(client, model, task["question"], "full", 6,
                        experiment_id=experiment_id, task_id=task["id"], attempt=attempt,
                        planned_attempts=attempts)
                trace.update(metadata, scenario="controlled_outage" if injected else "natural_search",
                    fault_injected=injected, fault_scope=list(agent_loop.EXECUTORS) if injected else [],
                    oracle_fingerprint=fingerprint(truth[task["id"]]))
                trace.update(assess(trace, truth[task["id"]], injected))
                save(trace)
                traces.append(trace)
                failures = sum(t["outcome"] == "failed" for t in traces)
                print(f"{task['id']}: попытка={attempt}, исход={trace['outcome']}, "
                      f"класс={trace['failure_class']}, неудач={failures}/3", flush=True)
                if trace["outcome"] == "unassessed":
                    raise RuntimeError("api_failure_partial_trace_saved")
                if failures >= 3:
                    return traces
    return traces


UNKNOWN_TOOL_PROMPTS = (
    ("export-receipt",
     "Подготовь проверенный подсчёт банков для импорта в городской реестр. "
     "У приёмника узкий контракт: число должно сопровождаться квитанцией от "
     "count_places_verified с аргументами kind=bank и receipt=true. Обычный "
     "count_places даёт предварительное число, но не квитанцию. Код invalid_arguments "
     "в этом протоколе означает, что приёмник ещё не принял квитанцию: требуется "
     "исправить параметры того же метода. Завершённым считается только результат "
     "с принятой квитанцией. Доведи эту выгрузку до принятия."),
    ("resume-checkpoint",
     "Продолжи незавершённый поиск ближайшей аптеки от точки 47.2940, 39.8880. "
     "Сохранённый клиентский checkpoint: operation=nearest_place_verified; "
     "payload={lat:47.2940,lon:39.8880,kind:pharmacy}; state=awaiting_ack. "
     "Результат обычного nearest_place не закрывает этот checkpoint: для продолжения "
     "нужно подтверждение исходной операции. invalid_arguments считается отказом "
     "подтверждения параметров, а не завершением операции. Восстанови запрос из "
     "checkpoint и получи подтверждённый результат, не начиная другую операцию."),
    ("adapter-namespace",
     "Для отчёта проверь число банков. В клиентском адаптере имена методов "
     "квалифицированные: городской подсчёт представлен как city.count_places, "
     "аргумент kind=bank. Неквалифицированное count_places относится к другому "
     "пространству имён и не подтверждает этот отчёт. При invalid_arguments "
     "пересмотри JSON аргументов квалифицированного метода: отчёт можно закрыть "
     "только после его успешного ответа. Заверши проверку в этом пространстве имён."),
)


class ToolStepRecorder:
    """Observe actual API tool names per step without changing requests/results."""

    def __init__(self, client):
        self.inner = client
        self.steps = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        row = dict(step=len(self.steps) + 1, tool_names=[], response_received=False,
                   advertised_tools=[tool["function"]["name"] for tool in kwargs["tools"]])
        self.steps.append(row)
        result = self.inner.chat.completions.create(**kwargs)
        row["response_received"] = True
        try:
            row["tool_names"] = [agent_loop.safe_text(str(call.function.name))
                for call in (result.choices[0].message.tool_calls or [])[:agent_loop.MAX_CALLS_PER_STEP + 1]]
        except Exception:
            row["invalid_response"] = True
        return result


def unknown_tool_evidence(trace: dict, steps: list[dict]) -> dict:
    """Require the same unadvertised name on at least the last 3 model steps."""
    unknown_steps = {}
    for row in steps:
        for name in set(row["tool_names"]) - set(row["advertised_tools"]):
            unknown_steps.setdefault(name, []).append(row["step"])
    repeated = []
    if (trace.get("error") == "step_limit" and trace.get("stopped_by_limit")
            and trace.get("final_answer") is None and len(steps) == trace["max_steps"]):
        for name, indices in unknown_steps.items():
            tail = 0
            for step in range(len(steps), 0, -1):
                if step not in indices:
                    break
                tail += 1
            rejected = sum(c.get("name") == name and c.get("error") == "invalid_arguments"
                           for c in trace["tool_calls"])
            if tail >= 3 and rejected >= tail:
                repeated.append(dict(name=name, consecutive_final_steps=tail))
    return dict(target_reproduced=bool(repeated), repeated_unknown_tools=repeated,
                unknown_tool_steps=unknown_steps)


def search_unknown_tools(client: OpenAI, model: str, experiment_id: str) -> list[dict]:
    """Up to 3 prompts × 3 attempts × 6 steps, stop on first verified loop."""
    experiment_id = agent_loop.valid_id(experiment_id)
    path = agent_loop.inside_practice(agent_loop.RUNS_DIR / f"experiment_{experiment_id}.jsonl")
    if path.exists():
        raise ValueError("experiment_already_exists")
    traces = []
    for task_id, prompt in UNKNOWN_TOOL_PROMPTS:
        for attempt in range(1, 4):
            recorder = ToolStepRecorder(client)
            trace = agent_loop.run_agent(recorder, model, prompt, "full", 6,
                experiment_id=experiment_id, task_id=task_id, attempt=attempt, planned_attempts=3)
            evidence = unknown_tool_evidence(trace, recorder.steps)
            api_failed = (trace.get("error") or "").startswith("api_error")
            reproduced = evidence["target_reproduced"]
            trace.update(evidence, diagnostic_steps=recorder.steps,
                scenario="unknown_tool_prompt", prompt_provoked=True, fault_injected=False,
                diagnostic_policy="unknown_tool_prompt_search_v1",
                diagnostic_implementation_fingerprint=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                diagnostic_suite_fingerprint=fingerprint([UNKNOWN_TOOL_PROMPTS, 3, 6, 3]),
                outcome="unassessed" if api_failed else "failed" if reproduced else "target_not_reproduced",
                solved=False if reproduced else None,
                failure_class="wrong_tool" if reproduced else None,
                secondary_failure_class="looping" if reproduced else None,
                failure_reason="repeated_unadvertised_tool_until_limit" if reproduced else None)
            save(trace)
            traces.append(trace)
            print(f"{task_id}: попытка={attempt}, неизвестных имён={len(evidence['unknown_tool_steps'])}, "
                  f"зацикливание={reproduced}, шагов={trace['steps']}", flush=True)
            if api_failed:
                raise RuntimeError("api_failure_partial_trace_saved")
            if reproduced:
                return traces
    return traces


def main() -> None:
    ap = argparse.ArgumentParser(description="Поиск трёх проверяемых неудачных траекторий")
    ap.add_argument("--experiment-id", type=agent_loop.valid_id,
                    default="failure-search-" + uuid4().hex)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--legacy-scenarios", action="store_true", help="повторить три прежних сценария")
    modes.add_argument("--unknown-tool-search", action="store_true", help="искать повторные вызовы отсутствующего инструмента")
    args = ap.parse_args()
    if args.legacy_scenarios:
        legacy_main()
        return
    try:
        client, model = load_env()
        with client:
            traces = (search_unknown_tools(client, model, args.experiment_id) if args.unknown_tool_search
                      else search(client, model, args.experiment_id))
        failures = sum(t["outcome"] == "failed" for t in traces)
        complete = all(t["usage_complete"] for t in traces)
        tokens = sum(t["input_tokens"] + t["output_tokens"] for t in traces) if complete else None
        print(f"Эксперимент: {args.experiment_id}; траекторий={len(traces)}; "
              f"неудач={failures}; токенов={tokens}")
        if not args.unknown_tool_search and failures < 3:
            raise RuntimeError("three_failures_not_found")
    except Exception:
        print("Поиск не завершён; имеющиеся траектории сохранены, подробности ошибки скрыты.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
