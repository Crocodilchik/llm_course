#!/usr/bin/env python3
"""Explicit one-request provider check; no config, responses or exceptions are dumped."""

import json

from agent_loop import load_client, make_tools, validate_args


def probe(client, model: str) -> bool:
    """Check the actual expected call and usage, without executing a tool."""
    try:
        response = client.chat.completions.create(
            model=model, tools=[make_tools("full")[0]],
            messages=[{"role": "user", "content": "Сколько в Ростове-на-Дону аптек?"}],
        )
        calls = response.choices[0].message.tool_calls or []
        if not calls or calls[0].function.name != "count_places":
            return False
        args = validate_args("count_places", json.loads(calls[0].function.arguments))
        usage = response.usage
        return (args.get("kind") == "pharmacy" and not args.get("only_24_7", False)
                and type(usage.prompt_tokens) is int and usage.prompt_tokens >= 0
                and type(usage.completion_tokens) is int and usage.completion_tokens >= 0)
    except Exception:
        return False


def main() -> int:
    try:
        client, model = load_client()
        with client:
            ok = probe(client, model)
    except Exception:
        ok = False
    print("Проверка tool_calls и usage: " + ("пройдена" if ok else "не пройдена"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
