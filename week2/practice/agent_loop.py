#!/usr/bin/env python3
"""Агент без фреймворка: модель в цикле с инструментами над базой Ростова.

Здесь нет openai-agents — только клиент openai (chat.completions) и собственный
цикл. Это то, что задание 1 требует собрать руками: инструменты, лимит шагов,
ошибки как наблюдения, журнал в JSONL и счётчики токенов.

Файл делится на четыре части:
1. РАСХОД — входные и выходные токены из usage, без валютного тарифа.
2. ИНСТРУМЕНТЫ — пять функций над базой + их схемы и два варианта описаний
   (short = одна фраза, full = что/когда/когда-НЕ/единицы/пример) для A/B.
3. ЦИКЛ — run_agent(): модель -> tool_calls -> исполнение -> наблюдение -> повтор.
4. ЖУРНАЛ — save_run(): каждая траектория пишется одной строкой JSON в runs/.

Запуск из week2/practice:  uv run python agent_loop.py "Сколько аптек в Ростове?"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from openai import OpenAI

# Чтобы `import sdk_tasks` работал независимо от того, откуда запущен скрипт,
# добавляем каталог этого файла (week2/practice) в начало sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sdk_tasks  # готовые функции над базой (даны курсом)
from answer_schema import FINAL_INSTRUCTION, FORMAT_VERSION, parse_answer


# ────────────────────────────────────────────────────────────────────────────
# 1. РАСХОД ТОКЕНОВ И БЕЗОПАСНАЯ ДИАГНОСТИКА
# ────────────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
MAX_RESULT_CHARS = 4_000
MAX_CALLS_PER_STEP = 10
SECRET_FIELDS = {"api_key", "openai_api_key", "hf_token", "password", "secret",
                 "authorization", "cookie", "access_token", "openai_base_url", "model"}


def safe_text(value: str) -> str:
    """Redact recognizable credentials, contacts and URLs without reading .env."""
    value = re.sub(r"(?i)-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*?-----END [^-]*PRIVATE KEY-----",
                   "[PRIVATE KEY REDACTED]", value)
    value = re.sub(r"(?i)\b(?:sk-[\w-]+|hf_[\w-]+|gh[pousr]_[\w-]+)", "[TOKEN REDACTED]", value)
    value = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", value)
    value = re.sub(r'''(?ix)(["']?(?:openai_api_key|api_key|password|secret|token|cookie|authorization)["']?\s*[:=]\s*)["']?[^\s,;\}\n]+''',
                   r"\1[REDACTED]", value)
    value = re.sub(r"https?://[^\s]+", "[URL REDACTED]", value)
    value = re.sub(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[EMAIL REDACTED]", value)
    return re.sub(r"(?<!\w)(?:\+?[78][\s()-]*)?(?:\d[\s()-]*){10}(?!\w)",
                  "[PHONE REDACTED]", value)


def safe_value(value):
    if isinstance(value, str):
        return safe_text(value)
    if isinstance(value, dict):
        return {safe_text(str(k)): (v if str(k).endswith("_fingerprint")
                                   and isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v)
                                   else safe_value(v))
                for k, v in value.items() if str(k).lower() not in SECRET_FIELDS}
    if isinstance(value, (list, tuple)):
        return [safe_value(v) for v in value]
    return value


def safe_error(exc: Exception, category: str) -> str:
    """Never serialize exception messages, traceback, request objects or config."""
    allowed = (ZeroDivisionError, TypeError, ValueError, KeyError, TimeoutError,
               ConnectionError, OSError)
    label = next((t.__name__ for t in allowed if isinstance(exc, t)), "Error")
    return f"{category}:{label}"


def inside_practice(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(HERE):
        raise ValueError("path_outside_practice")
    return resolved


def valid_id(value: str) -> str:
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value)
            or re.match(r"(?i)^(?:sk-|hf_|gh[pousr]_)", value)):
        raise ValueError("invalid_identifier")
    return value


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("ожидается положительное целое число")
    return result


def load_client() -> tuple[OpenAI, str]:
    """Application-only config loading. No config values are printed or logged."""
    from dotenv import load_dotenv
    # Only this directory and the known repository root; never arbitrary cwd ancestors.
    root = HERE.parent.parent if (HERE.parent / "assignment.md").is_file() else HERE.parent
    for path in (HERE / ".env", root / ".env"):
        if path.is_file():
            load_dotenv(path)
            break
    values = {name: os.getenv(name, "").strip()
              for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "MODEL")}
    if not all(values.values()):
        raise ValueError("missing_configuration")
    # No implicit retries: a network failure must not hide additional requests.
    return (OpenAI(api_key=values["OPENAI_API_KEY"], base_url=values["OPENAI_BASE_URL"],
                   timeout=30.0, max_retries=0), values["MODEL"])


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
    address = " ".join(str(card[k]) for k in ("street", "housenumber") if card.get(k))
    return (f"{card['name'] or 'без названия'} · адрес {address or 'нет данных'} · "
            f"часы {card['opening_hours'] or 'нет данных'} · "
            f"телефон [контакт скрыт] (id={card['id']})")


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
            "street": {"type": ["string", "null"],
                       "description": "часть названия улицы; без неё ищется по всему городу"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50,
                      "description": "сколько строк вернуть (1–50)"},
        },
    },
    "nearest_place": {
        "type": "object",
        "required": ["lat", "lon", "kind"],
        "properties": {
            "lat": {"type": "number", "minimum": -90, "maximum": 90, "description": "широта точки в градусах"},
            "lon": {"type": "number", "minimum": -180, "maximum": 180, "description": "долгота точки в градусах"},
            "kind": {"type": "string",
                     "description": "вид места латиницей: pharmacy, cafe, outpost, ..."},
        },
    },
    "place_details": {
        "type": "object",
        "required": ["place_id"],
        "properties": {
            "place_id": {"type": "integer", "minimum": 1, "description": "идентификатор места"},
        },
    },
    "top_brands": {
        "type": "object",
        "required": ["kind"],
        "properties": {
            "kind": {"type": "string",
                     "description": "вид места латиницей: pharmacy, outpost, ..."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50,
                      "description": "сколько брендов вернуть (1–50)"},
        },
    },
}


# Advertise the same constraints that the local validator enforces.
for _schema in SCHEMAS.values():
    _schema["additionalProperties"] = False
    for _field, _spec in _schema["properties"].items():
        if _spec.get("type") == "string" or _spec.get("type") == ["string", "null"]:
            _spec.update(minLength=1, maxLength=120)
        if _field == "kind":
            _spec.update(maxLength=64, pattern=r"^[a-z][a-z0-9_]{0,63}$")


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
            "Единицы: place_id — целое число из ответа другого инструмента. "
            "Пример: «Адрес места с id 42» → place_id=42. Контакт скрывается."
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
            "name": name, "description": descs[name],
            "parameters": {**SCHEMAS[name], "additionalProperties": False},
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


def validate_args(name: str, args: dict) -> dict:
    """Validate exactly the advertised schema before executing any function."""
    if name not in EXECUTORS or name not in SCHEMAS or not isinstance(args, dict):
        raise ValueError("invalid_tool_or_arguments")
    schema = SCHEMAS[name]
    props = schema.get("properties", {})
    if set(args) - set(props) or set(schema.get("required", [])) - set(args):
        raise ValueError("unexpected_or_missing_arguments")
    for key, value in args.items():
        spec = props[key]
        types = spec.get("type")
        types = types if isinstance(types, list) else [types]
        correct = ((value is None and "null" in types)
                   or (type(value) is bool and "boolean" in types)
                   or (type(value) is int and "integer" in types)
                   or (type(value) in (int, float) and "number" in types and math.isfinite(value))
                   or (isinstance(value, str) and "string" in types and 0 < len(value) <= 120))
        if not correct:
            raise ValueError("argument_type")
        if isinstance(value, str) and key == "kind" and not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value):
            raise ValueError("argument_kind")
        if type(value) in (int, float):
            if value < spec.get("minimum", -math.inf) or value > spec.get("maximum", math.inf):
                raise ValueError("argument_bounds")
    return args


def _fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def protocol_metadata(model: str, variant: str) -> dict:
    """Fingerprints identify config without exposing environment values."""
    files = ("agent_loop.py", "answer_schema.py", "check.py", "sdk_tasks.py")
    code = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in files}
    return {"model_fingerprint": _fingerprint(model),
            "implementation_fingerprint": _fingerprint(code),
            "protocol_fingerprint": _fingerprint([SYSTEM, FINAL_INSTRUCTION, SCHEMAS]),
            "descriptions_fingerprint": _fingerprint(DESCRIPTIONS[variant])}


def run_agent(client: OpenAI, model: str, question: str, variant: str,
              max_steps: int = 6, *, experiment_id: str | None = None,
              task_id: str | None = None, attempt: int | None = None,
              planned_attempts: int | None = None) -> dict:
    """One isolated attempt; API errors return a partial trace, never raw exceptions."""
    if type(max_steps) is not int or not 1 <= max_steps <= 32:
        raise ValueError("max_steps_must_be_1_to_32")
    if not isinstance(question, str) or not question.strip() or len(question) > 4_000:
        raise ValueError("invalid_question")
    if attempt is not None and (type(attempt) is not int or attempt < 1):
        raise ValueError("invalid_attempt")
    tools = make_tools(variant)
    messages = [{"role": "system", "content": SYSTEM + "\n" + FINAL_INSTRUCTION},
                {"role": "user", "content": safe_text(question)}]
    trace = {
        "format_version": FORMAT_VERSION, "trace_id": uuid4().hex,
        "experiment_id": valid_id(experiment_id or uuid4().hex),
        "task_id": valid_id(task_id) if task_id else None, "attempt": attempt,
        "planned_attempts": planned_attempts,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": safe_text(question), "variant": variant, "max_steps": max_steps,
        "steps": 0, "tool_calls": [], "final_answer": None,
        "input_tokens": 0, "output_tokens": 0,
        "known_input_tokens": 0, "known_output_tokens": 0,
        "usage_complete": True, "error": None, "stopped_by_limit": False,
        "answer_errors": 0, **protocol_metadata(model, variant),
    }
    for step in range(1, max_steps + 1):
        trace["steps"] = step
        try:
            resp = client.chat.completions.create(model=model, messages=messages, tools=tools)
        except Exception as exc:
            trace.update(error=safe_error(exc, "api_error"), usage_complete=False,
                         input_tokens=None, output_tokens=None)
            return trace
        usage = getattr(resp, "usage", None)
        ins = getattr(usage, "prompt_tokens", None)
        outs = getattr(usage, "completion_tokens", None)
        if type(ins) is int and ins >= 0 and type(outs) is int and outs >= 0:
            trace["known_input_tokens"] += ins
            trace["known_output_tokens"] += outs
            if trace["usage_complete"]:
                trace["input_tokens"] += ins
                trace["output_tokens"] += outs
        else:
            trace.update(usage_complete=False, input_tokens=None, output_tokens=None)
        try:
            msg = resp.choices[0].message
            calls = msg.tool_calls or []
            if len(calls) > MAX_CALLS_PER_STEP:
                raise ValueError("too_many_calls")
            serialized = [{"id": tc.id, "type": "function", "function": {
                "name": tc.function.name, "arguments": safe_text(tc.function.arguments or "{}")}}
                for tc in calls]
        except Exception:
            trace["error"] = "invalid_response"
            return trace
        if not calls:
            try:
                answer = parse_answer(msg.content)
                trace["final_answer"] = safe_value(answer)
                return trace
            except Exception:
                trace["answer_errors"] += 1
                # Do not retain an invalid, potentially sensitive final response.
                messages.append({"role": "user", "content":
                                 "Финальный ответ не прошёл проверку JSON. " + FINAL_INSTRUCTION})
                continue
        messages.append({"role": "assistant", "content": None, "tool_calls": serialized})
        for tc in calls:
            name = tc.function.name
            args = None
            try:
                if len(tc.function.arguments or "{}") > 4_000:
                    raise ValueError("arguments_too_large")
                args = validate_args(name, json.loads(tc.function.arguments or "{}"))
            except Exception:
                result, err = "ОШИБКА: invalid_arguments", "invalid_arguments"
            else:
                try:
                    if not sdk_tasks.DB.is_file():
                        raise OSError("database_missing")
                    result = EXECUTORS[name](**args)
                    if not isinstance(result, str):
                        raise TypeError("result_must_be_text")
                    result = safe_text(result)
                    if len(result) > MAX_RESULT_CHARS:
                        result, err = "ОШИБКА: result_too_large; уменьшите limit", "result_too_large"
                    else:
                        err = None
                except Exception as exc:
                    err = safe_error(exc, "tool_error")
                    result = "ОШИБКА: " + err
            trace["tool_calls"].append({"name": safe_text(name), "args": safe_value(args),
                                        "result": result, "error": err})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    trace["stopped_by_limit"] = True
    trace["error"] = "step_limit"
    return trace


# ────────────────────────────────────────────────────────────────────────────
# 4. ЖУРНАЛ
# ────────────────────────────────────────────────────────────────────────────
RUNS_DIR = Path(__file__).with_name("runs")


def save_run(trace: dict) -> Path:
    """Append only to this experiment; reject links pointing outside PRACTICE."""
    directory = inside_practice(RUNS_DIR)
    directory.mkdir(exist_ok=True)
    path = inside_practice(directory / f"experiment_{valid_id(trace['experiment_id'])}.jsonl")
    payload = safe_value(trace)
    # Controlled identifiers are not free text: a date like 20261021-01 is not
    # a phone number. Preserve only validated top-level IDs; nested user content
    # and recognizable credential prefixes still undergo normal sanitization.
    for field in ("trace_id", "experiment_id", "task_id"):
        if trace.get(field) is not None:
            payload[field] = valid_id(trace[field])
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    return path


def main() -> None:
    """CLI: one question, typed answer, safe status and token consumption."""
    ap = argparse.ArgumentParser(description="Агент без фреймворка над базой Ростова")
    ap.add_argument("question", help="вопрос агенту")
    ap.add_argument("--variant", choices=("short", "full"), default="full",
                    help="вариант описаний инструментов (A/B)")
    ap.add_argument("--max-steps", type=positive_int, default=6, help="лимит шагов (1–32)")
    ap.add_argument("--experiment-id", type=valid_id, default=None)
    args = ap.parse_args()

    if args.max_steps > 32:
        ap.error("лимит шагов должен быть 1–32")
    try:
        client, model = load_client()
        with client:
            trace = run_agent(client, model, args.question, args.variant, args.max_steps,
                              experiment_id=args.experiment_id)
        save_run(trace)
    except Exception:
        print("Ошибка настройки, аргументов или сохранения; подробности скрыты.")
        raise SystemExit(1) from None
    print("Ответ:", json.dumps(trace["final_answer"], ensure_ascii=False))
    print(f"Шагов: {trace['steps']}; входных токенов: {trace['input_tokens']}; "
          f"выходных: {trace['output_tokens']}; ошибка: {trace['error']}")
    if trace["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
