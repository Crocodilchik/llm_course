# Сторонние материалы

Лицензия MIT в `LICENSE` относится к собственным дополнениям участников
проекта. Она не изменяет права на перечисленные ниже сторонние материалы.

## OpenStreetMap

© OpenStreetMap contributors. `week2/practice/data/rostov_poi.json` — снимок
OpenStreetMap по Ростову-на-Дону; `week2/practice/poi.sqlite` — база на его основе.
Условия базы: Open Data Commons Open Database License (ODbL) 1.0.
При распространении базы сохраняйте атрибуцию и соблюдайте условия ODbL.

- [Авторство и условия OpenStreetMap](https://www.openstreetmap.org/copyright).
- [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
- [Database Contents License 1.0](https://opendatacommons.org/licenses/dbcl/1-0/).

## Учебные материалы MMCS

`week2/assignment.md`, исходная обвязка `week2/practice/sdk_tasks.py`, исходные
части `week2/practice/make_poi_dataset.py` и иллюстрации `week2/practice/img/`
получены с материалами курса «Агентный ИИ: теория и инженерия» MMCS.
Авторство сохраняется за их правообладателями. Явная свободная лицензия на
эти материалы в проверенном локальном дереве не найдена; MIT проекта на них
не распространяется. Перед публичным размещением следует подтвердить
разрешение правообладателей на распространение материалов курса.

## Зависимости и модели

Python-пакеты, PostgreSQL, pgvector и компоненты контейнерных образов сохраняют
свои лицензии. Версии Python-зависимостей закреплены в двух `uv.lock`, образы —
в Dockerfile и Compose. Зависимости и веса не переоформляются под MIT проекта.

Векторизация использует `intfloat/multilingual-e5-small`, закреплённую на
revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`. Модель загружается отдельно;
её авторство, лицензия и ограничения указаны в
[карточке модели](https://huggingface.co/intfloat/multilingual-e5-small).
Условия доступа к внешней языковой модели определяются её провайдером.
