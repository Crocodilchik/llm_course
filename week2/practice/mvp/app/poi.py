"""Read-only public POI access. No process-global call log or contact fields."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import aiosqlite
from .schemas import clean_text

FOOD_KINDS = ("restaurant", "cafe", "fast_food")
FIELDS = "id, kind, name, brand, street, housenumber, opening_hours, cuisine, lat, lon"


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
    return int(math.hypot((lat1-lat2)*111_320,
                         (lon1-lon2)*111_320*math.cos(math.radians(47.23))))


def public(row) -> dict:
    return {k: clean_text(v) if isinstance(v, str) else v for k, v in dict(row).items()}


class PoiRepository:
    def __init__(self, path: Path):
        self.path = path.resolve()

    async def rows(self, sql: str, parameters=()) -> list[dict]:
        async with aiosqlite.connect(self.path.as_uri() + "?mode=ro&immutable=1", uri=True) as con:
            con.row_factory = aiosqlite.Row
            async with con.execute(sql, parameters) as cursor:
                return [public(row) for row in await cursor.fetchall()]

    async def count(self, kind: str, only_24_7: bool = False) -> dict:
        rows = await self.rows("SELECT COUNT(*) AS count FROM poi WHERE kind=?" +
                              (" AND is_24_7=1" if only_24_7 else ""), (kind,))
        return rows[0]

    async def find(self, kind: str, street: str | None = None, limit: int = 5) -> list[dict]:
        sql, params = f"SELECT {FIELDS} FROM poi WHERE kind=?", [kind]
        if street:
            sql += " AND street LIKE ? ESCAPE '\\'"
            literal = street.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params.append("%" + literal + "%")
        return await self.rows(sql + " ORDER BY id LIMIT ?", (*params, limit))

    async def nearest(self, lat: float, lon: float, kind: str) -> dict | None:
        rows = await self.rows(f"SELECT {FIELDS} FROM poi WHERE kind=? ORDER BY id", (kind,))
        if not rows:
            return None
        place = min(rows, key=lambda row: distance_m(lat, lon, row["lat"], row["lon"]))
        return {**place, "distance_m": distance_m(lat, lon, place["lat"], place["lon"])}

    async def details(self, place_id: int) -> dict | None:
        rows = await self.rows(f"SELECT {FIELDS} FROM poi WHERE id=?", (place_id,))
        return rows[0] if rows else None

    async def brands(self, kind: str, limit: int) -> list[dict]:
        return await self.rows("SELECT COALESCE(brand, 'без бренда') AS brand, COUNT(*) AS count "
            "FROM poi WHERE kind=? GROUP BY 1 ORDER BY count DESC, brand LIMIT ?", (kind, limit))

    async def food(self) -> list[dict]:
        return await self.rows(f"SELECT {FIELDS} FROM poi WHERE kind IN (?,?,?) ORDER BY id", FOOD_KINDS)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()
