"""Maintenance entry point. Never print configuration or generated secrets."""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import secrets


def prepare(root: Path | None = None) -> None:
    root = (root or Path(__file__).resolve().parents[2]).resolve()
    for relative in ("private/mvp/app", "private/mvp/postgres", ".cache/models"):
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError("unsafe_runtime_path")
        target.mkdir(parents=True, exist_ok=True)
    password = (root / "private/mvp/app/db_password").resolve()
    if not password.is_relative_to(root):
        raise ValueError("unsafe_secret_path")
    try:
        with password.open("x", encoding="utf-8") as stream:
            stream.write(secrets.token_urlsafe(40))
        os.chmod(password, 0o600)
    except FileExistsError:
        pass


async def build_index() -> None:
    from .config import Settings
    from .db import Database
    from .embeddings import Embedder
    from .poi import PoiRepository
    from .search import SearchService
    settings = Settings.load()
    db = Database(settings.database_url)
    try:
        await db.initialize()
        embedder = Embedder(settings)
        service = SearchService(db, PoiRepository(settings.poi_path), embedder)
        result = await service.build()
        print(f"Индекс готов: документов={result['documents']}; версия={result['version']}")
    finally:
        if 'embedder' in locals():
            embedder.close()
        await db.close()


async def evaluate_search() -> None:
    import json
    from .config import Settings
    from .db import Database
    from .embeddings import Embedder
    from .poi import PoiRepository
    from .search import SearchService,DATA
    settings=Settings.load();db=Database(settings.database_url);embedder=Embedder(settings)
    poi=PoiRepository(settings.poi_path);service=SearchService(db,poi,embedder)
    try:
        index=await service.build();cases=json.loads((DATA/'semantic_cases.json').read_text(encoding='utf-8'))
        places=await poi.food();rows=[]
        def hit(results,case):
            if case.get('expected_empty'):return not results
            expected=set(case['cuisines'])
            return any(expected.intersection((p.get('cuisine') or '').replace(',',';').split(';')) for p in results[:5])
        for case in cases:
            results=await service.query(case['query'])
            literal=[p for p in places if case['query'].lower() in ((p.get('name') or '')+' '+(p.get('cuisine') or '')).lower()][:5]
            rows.append({**case,'hit':hit(results,case),'literal_hit':hit(literal,case),
                'results':[{'id':p['id'],'cuisine':p.get('cuisine'),'similarity':p['similarity']} for p in results]})
        positive=[r for r in rows if not r.get('expected_empty')];negative=[r for r in rows if r.get('expected_empty')]
        result={'index':index,'threshold':0.84,'positive_hit_at_5':sum(r['hit'] for r in positive)/len(positive),
                'literal_positive_hit_at_5':sum(r['literal_hit'] for r in positive)/len(positive),
                'negative_empty':sum(r['hit'] for r in negative),'negative_total':len(negative),
                'calibration_set':True,'cases':rows}
        target=settings.private_path/'semantic-evaluation.json'
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print('SAFE:'+json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        embedder.close();await db.close()


def main() -> None:
    import logging
    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "index", "evaluate-search", "serve"))
    action = parser.parse_args().action
    try:
        if action == "prepare":
            prepare()
            print("Каталоги и локальный секрет БД подготовлены; значения скрыты.")
        elif action == "index":
            asyncio.run(build_index())
        elif action == "evaluate-search":
            asyncio.run(evaluate_search())
        else:
            import uvicorn
            uvicorn.run("app.main:app", host="0.0.0.0", port=8000,
                        access_log=False, log_level="warning", workers=1)
    except Exception:
        print("Операция не завершена; проверьте настройки и доступность сервисов. Подробности скрыты.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
