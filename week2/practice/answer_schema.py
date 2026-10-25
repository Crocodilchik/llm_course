"""Typed final answers; deterministic validation, never an LLM-as-judge."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

FORMAT_VERSION = 2


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class PlaceRef(StrictModel):
    id: int = Field(gt=0)
    name: str | None = Field(default=None, max_length=200)


class CountAnswer(StrictModel):
    type: Literal["count"]
    count: int = Field(ge=0)


class PlacesAnswer(StrictModel):
    type: Literal["places"]
    places: list[PlaceRef] = Field(min_length=1, max_length=50)


class NearestAnswer(StrictModel):
    type: Literal["nearest"]
    place: PlaceRef
    distance_m: int = Field(ge=0)


class BrandsAnswer(StrictModel):
    type: Literal["brands"]
    brands: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        min_length=1, max_length=50
    )


class DetailsAnswer(StrictModel):
    type: Literal["details"]
    place: PlaceRef
    address: str | None = Field(default=None, max_length=300)
    opening_hours: str | None = Field(default=None, max_length=200)


class UnavailableAnswer(StrictModel):
    type: Literal["unavailable"]
    reason: Literal["no_data", "not_found", "unsupported"]


Answer = Annotated[
    CountAnswer | PlacesAnswer | NearestAnswer | BrandsAnswer | DetailsAnswer | UnavailableAnswer,
    Field(discriminator="type"),
]
ADAPTER = TypeAdapter(Answer)

# One identical contract for both description variants. No sixth tool is added.
FINAL_INSTRUCTION = """
Финальный ответ — только один JSON-объект, без markdown и свободного текста.
Выбери одну форму по вопросу:
{"type":"count","count":19}
{"type":"places","places":[{"id":1,"name":"Название"}]}
{"type":"nearest","place":{"id":1,"name":"Название"},"distance_m":100}
{"type":"brands","brands":["Название сети"]}
{"type":"details","place":{"id":1,"name":"Название"},"address":null,"opening_hours":null}
{"type":"unavailable","reason":"no_data"}
reason также может быть not_found или unsupported. Используй unavailable, когда
база не позволяет ответить (например, выбрать экономически лучшее место).
Числа, id и названия бери только из наблюдений инструментов. Не добавляй поля.
Числа в примерах — только примеры формата, а не ответы на текущий вопрос.
""".strip()


def parse_answer(value: str | dict) -> dict:
    """Reject invalid JSON, extra fields, booleans as integers and non-finite values."""
    if isinstance(value, str):
        if len(value) > 16_000:
            raise ValueError("answer_too_large")
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("answer_must_be_object")
    return ADAPTER.validate_python(value).model_dump()
