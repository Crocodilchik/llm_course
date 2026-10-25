from app.search import SearchService
from app.poi import PoiRepository
from conftest import FakeEmbedder

async def test_index_version_idempotent_and_street(app,settings):
    search=SearchService(app.state.db,PoiRepository(settings.poi_path),FakeEmbedder(settings))
    first=await search.build();second=await search.build()
    assert first==second and first['documents']==604
    assert await search.query('пельмени',street='несуществующая-улица')==[]
    results=await search.query('кафе',limit=3)
    assert len(results)==3 and all('меню не подтверждено' in p['evidence'] for p in results)

async def test_index_failure_preserves_active_version(app,settings):
    poi=PoiRepository(settings.poi_path)
    search=SearchService(app.state.db,poi,FakeEmbedder(settings));first=await search.build()
    class Broken(FakeEmbedder):
        async def encode(self,*args,**kwargs):raise RuntimeError('fixture outage')
    other=SearchService(app.state.db,poi,Broken(settings));other.aliases['fixture']='changed'
    import pytest
    with pytest.raises(RuntimeError):await other.build()
    assert await other.active()==first and other.state=='failed'
