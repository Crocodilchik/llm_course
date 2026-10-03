#!/usr/bin/env python3
"""Проверка сервиса: один запрос к модели через OpenAI-совместимый API.

Проверяет два свойства, без которых практика не поедет (см. week1/setup.md):
1. модель умеет вызывать инструменты в формате tool_calls;
2. в ответе приходит usage (счётчики токенов).

Переменные берутся из .env: OPENAI_BASE_URL, OPENAI_API_KEY, MODEL.
Ключ читается только из окружения/.env и нигде не печатается.

Запуск (из корня репозитория):
    uv run python week2/practice/check_provider.py
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    sys.exit("Нужен python-dotenv: uv add python-dotenv")

# Устойчивый вывод кириллицы в консоли Windows.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")


def find_env_file() -> Path | None:
    """Ищем .env, поднимаясь от каталога скрипта вверх до корня репозитория."""
    here = Path(__file__).resolve().parent
    for d in (here, *here.parents):
        env = d / ".env"
        if env.exists():
            return env
    return None


def main() -> int:
    env_file = find_env_file()
    if env_file:
        load_dotenv(env_file)
    else:
        print("Предупреждение: .env не найден — читаю переменные окружения.")

    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("MODEL", "").strip()

    missing = [
        name
        for name, value in (
            ("OPENAI_BASE_URL", base_url),
            ("OPENAI_API_KEY", api_key),
            ("MODEL", model),
        )
        if not value
    ]
    if missing:
        sys.exit("Ошибка: в .env не заполнено: " + ", ".join(missing))

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    tool = {
        "type": "function",
        "function": {
            "name": "count_places",
            "description": "Сколько в городе мест такого вида. Возвращает одно число.",
            "parameters": {
                "type": "object",
                "required": ["kind"],
                "properties": {
                    "kind": {
                        "type": "string",
                        "description": "вид места латиницей: pharmacy, cafe, outpost",
                    }
                },
            },
        },
    }

    messages = [
        {
            "role": "system",
            "content": (
                "Ты агент, отвечающий на вопросы о городе. Используй переданные "
                "инструменты, чтобы получить точные данные."
            ),
        },
        {"role": "user", "content": "Сколько в Ростове-на-Дону аптек?"},
    ]

    print(f"Модель: {model}")
    print(f"Базовый URL: {base_url}")
    print("Отправляю один запрос с инструментом...")

    try:
        resp = client.chat.completions.create(model=model, tools=[tool], messages=messages)
    except Exception as exc:  # noqa: BLE001 — показываем причину, не раскрывая ключ
        print(f"\nОшибка запроса: {type(exc).__name__}: {exc}")
        return 1

    message = resp.choices[0].message

    # 1. tool_calls
    tool_calls_ok = False
    if message.tool_calls:
        call = message.tool_calls[0]
        name = getattr(call.function, "name", None)
        args = getattr(call.function, "arguments", None)
        if name and args is not None:
            tool_calls_ok = True
            print(f"\n[OK] tool_calls: модель попросила вызвать {name}({args})")
        else:
            print("\n[FAIL] tool_calls: вызов есть, но не хватает имени/аргументов")
    else:
        print(
            f"\n[FAIL] tool_calls: модель ответила текстом вместо вызова инструмента: "
            f"{message.content!r}"
        )

    # 2. usage
    usage_ok = False
    usage = resp.usage
    if usage is not None:
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        if prompt is not None and completion is not None:
            usage_ok = True
            total = getattr(usage, "total_tokens", "?")
            print(f"[OK] usage: вход {prompt}, выход {completion}, всего {total}")
        else:
            print("[FAIL] usage: объект usage есть, но без prompt_tokens/completion_tokens")
    else:
        print("[FAIL] usage: в ответе нет usage")

    print()
    if tool_calls_ok and usage_ok:
        print("ИТОГ: сервис готов — инструменты вызываются, usage приходит.")
        return 0
    print("ИТОГ: сервис НЕ готов. Смотрите ошибки выше.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
