import asyncio
from httpx import AsyncClient,ASGITransport
from conftest import register,send,wait_run

async def test_five_users_three_active(app):
    clients=[AsyncClient(transport=ASGITransport(app=app),base_url='http://testserver') for _ in range(5)]
    try:
        chats=[await register(client,'parallel_'+str(i)) for i,client in enumerate(clients)]
        app.state.fake.hold=True
        runs=await asyncio.gather(*(send(c,ch) for c,ch in zip(clients,chats)))
        for _ in range(100):
            if app.state.fake.active==3:break
            await asyncio.sleep(.01)
        assert app.state.fake.active==3
        for c in clients:assert (await c.get('/health')).status_code==200
        app.state.fake.release.set()
        results=await asyncio.gather(*(wait_run(c,r) for c,r in zip(clients,runs)))
        assert all(r['status']=='completed' for r in results)
        assert app.state.fake.peak==3
    finally:
        for client in clients:await client.aclose()
