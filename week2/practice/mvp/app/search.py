from __future__ import annotations
import asyncio
import hashlib
import json
import math
from pathlib import Path
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from .models import SearchDocument, Metadata

DATA = Path(__file__).resolve().parents[1] / "data"


class SearchService:
    def __init__(self, db, poi, embedder):
        self.db, self.poi, self.embedder = db, poi, embedder
        self.aliases = json.loads((DATA / "cuisine_aliases.json").read_text(encoding="utf-8"))
        self.state = "preparing"
        self.lock = asyncio.Lock()

    def version(self) -> str:
        config = getattr(self.embedder, "settings", None)
        fields = [self.poi.fingerprint(), self.aliases, "food-document-v2",
                  getattr(config, "embedding_model", "fixture"), getattr(config, "embedding_revision", "fixture")]
        return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def document(self, place: dict) -> str:
        kind = {"restaurant": "ресторан", "cafe": "кафе", "fast_food": "быстрое питание"}[place["kind"]]
        cuisines = (place.get("cuisine") or "").replace(",", ";").split(";")
        translated = " ".join(self.aliases.get(c.strip().lower(), c.strip()) for c in cuisines)
        return ". ".join(x for x in (place.get("name"), place.get("brand"), kind,
                                      place.get("cuisine"), translated) if x)

    async def active(self) -> dict | None:
        async with self.db.sessions() as session:
            metadata = await session.get(Metadata, "search_active")
            return metadata.value if metadata else None

    async def build(self) -> dict:
        async with self.lock:
            version = await asyncio.to_thread(self.version)
            current = await self.active()
            if current and current["version"] == version:
                self.state = "ready"
                return current
            self.state = "preparing"
            try:
                places = await self.poi.food()
                documents = [self.document(p) for p in places]
                vectors = await self.embedder.encode(documents)
                if len(vectors) != len(places) or not places:
                    raise ValueError("incomplete_index")
                async with self.db.sessions() as session:
                    dialect = self.db.engine.dialect.name
                    insert = pg_insert if dialect == "postgresql" else sqlite_insert
                    for offset in range(0, len(places), 32):
                        values = [dict(poi_id=p["id"], version=version, document=d, embedding=v, attributes=p)
                                  for p, d, v in zip(places[offset:offset+32], documents[offset:offset+32], vectors[offset:offset+32])]
                        await session.execute(insert(SearchDocument).values(values).on_conflict_do_nothing())
                    count = await session.scalar(select(func.count()).select_from(SearchDocument).where(SearchDocument.version == version))
                    if count != len(places):
                        raise ValueError("incomplete_index")
                    result = dict(version=version, documents=count)
                    existing = await session.get(Metadata, "search_active")
                    if existing:
                        existing.value = result
                    else:
                        session.add(Metadata(key="search_active", value=result))
                    await session.commit()
                self.state = "ready"
                return result
            except Exception:
                self.state = "failed"
                raise

    async def query(self, query: str, street: str | None = None, limit: int = 5) -> list[dict]:
        current = await self.active()
        if not current:
            raise RuntimeError("index_not_ready")
        expanded = query
        for key, value in self.aliases.items():
            if key.startswith("query:") and key[6:] in query.lower():
                expanded += ". " + value
        vector = (await self.embedder.encode([expanded], query=True))[0]
        async with self.db.sessions() as session:
            if self.db.engine.dialect.name == "postgresql":
                distance = SearchDocument.embedding.cosine_distance(vector).label("distance")
                statement = select(SearchDocument, distance).where(SearchDocument.version == current["version"])
                if street:
                    statement = statement.where(SearchDocument.attributes["street"].as_string().ilike("%"+street+"%"))
                rows = (await session.execute(statement.order_by(distance, SearchDocument.poi_id).limit(20))).all()
            else:
                # Fixture-only SQLite backend; shipped Docker uses real pgvector.
                docs = (await session.scalars(select(SearchDocument).where(SearchDocument.version == current["version"]))).all()
                rows = []
                for doc in docs:
                    if street and street.lower() not in (doc.attributes.get("street") or "").lower():
                        continue
                    denom = math.sqrt(sum(x*x for x in vector)*sum(x*x for x in doc.embedding))
                    dist = 1 - sum(x*y for x,y in zip(vector,doc.embedding))/denom if denom else 1
                    rows.append((doc, dist))
                rows = sorted(rows, key=lambda pair: (pair[1], pair[0].poi_id))[:20]
        result = []
        for document, distance in rows:
            score = 1 - float(distance)
            if query.lower() in (document.attributes.get("name") or "").lower():
                score += 0.03
            if score < 0.84:
                continue
            result.append({**document.attributes, "similarity": round(min(score,1),4),
                "evidence": "Название и теги кухни OSM; блюдо в меню не подтверждено"})
        return sorted(result, key=lambda p: (-p["similarity"], p["id"]))[:limit]
