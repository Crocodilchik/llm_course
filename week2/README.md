# Неделя 2. Задание 1 — инструменты и цикл без фреймворка

Агент над базой мест Ростова-на-Дону (8860 точек OpenStreetMap, ODbL), собранный
без фреймворка openai-agents: только клиент `openai` и собственный цикл.

## Требования

- Python 3.12+, менеджер зависимостей `uv`.
- `.env` в каталоге `week2/` (шаблон — `.env.example`):

```
OPENAI_BASE_URL=https://polza.ai/api/v1
OPENAI_API_KEY=<ваш ключ>
MODEL=deepseek/deepseek-v4.1-flash@provider=deepinfra/fp8
```

## Запуск на чистой машине (одной командой)

```bash
cd week2
uv sync
uv run python practice/check.py --n 3
```

`uv sync` ставит зависимости и создаёт окружение; `check.py --n 3` прогоняет десять
задач по три раза и печатает метрики (pass@1, pass^3, средние токены, стоимость, шаги).

## Команды

| Команда | Что делает |
|---|---|
| `uv run python practice/check.py --n 3` | полный прогон (вариант описаний `full`) |
| `uv run python practice/check.py --n 3 --variant short` | A/B-прогон с короткими описаниями |
| `uv run python practice/ab_report.py` | A/B-отчёт: доля верного выбора + интервал Вильсона |
| `uv run python practice/agent_loop.py "вопрос"` | один прогон агента |
| `uv run python practice/failures.py` | генерация трёх неудачных траекторий |

## Файлы

- `practice/agent_loop.py` — инструменты, цикл, лимит, журнал JSONL, стоимость.
- `practice/check.py` — 10 задач и метрики.
- `practice/ab_report.py` — A/B описаний.
- `practice/failures.py` — сценарии ошибок для разбора.
- `practice/report.md` — отчёт задания 1.
- `practice/week02_agents.ipynb` — практика недели 2 (11 задач на openai-agents SDK).
- `practice/sdk_tasks.py`, `practice/poi.sqlite`, `practice/data/` — функции и данные базы.
