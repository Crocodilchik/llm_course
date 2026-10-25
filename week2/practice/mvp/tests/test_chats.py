from httpx import AsyncClient,ASGITransport
from conftest import register,send,wait_run

async def test_idempotency_history_and_context(client,app):
    chat=await register(client);run=await send(client,chat)
    assert (await wait_run(client,run))['status']=='completed'
    assert await send(client,chat)==run
    second=await send(client,chat,key='request_0002',text='А ближайший к тем же координатам?')
    await wait_run(client,second)
    assert app.state.fake.calls==2
    assert app.state.fake.snapshots[-1][2]['filters']['lat']==47.23
    assert app.state.fake.snapshots[-1][2]['place_ids']==[1]
    assert len((await client.get('/api/chats/'+chat)).json()['runs'])==2

async def test_other_user_cannot_access(client,app):
    chat=await register(client);run=await send(client,chat);await wait_run(client,run)
    async with AsyncClient(transport=ASGITransport(app=app),base_url='http://testserver') as other:
        await register(other,'tester_two')
        for path in ['/api/chats/'+chat,'/api/runs/'+run,'/api/runs/'+run+'/events']:
            assert (await other.get(path)).status_code==404
        assert (await other.post('/api/runs/'+run+'/cancel')).status_code==404
        assert (await other.post('/api/chats/'+chat+'/messages',json={'text':'test','request_key':'request_other'})).status_code==404

async def test_cancel_and_active_user_limit(client,app):
    chat=await register(client);app.state.fake.hold=True;run=await send(client,chat)
    assert (await client.post('/api/chats/'+chat+'/messages',json={'text':'again','request_key':'request_0002'})).status_code==409
    assert (await client.post('/api/runs/'+run+'/cancel')).status_code==200
    assert (await wait_run(client,run))['status']=='cancelled'
    app.state.fake.hold=False
    second=await send(client,chat,'request_0003')
    assert (await wait_run(client,second))['status']=='completed'

async def test_restart_recovers_stale_run_without_model(settings):
    from app.main import create_app
    from app.models import Run,Chat
    from conftest import FakeRuntime,FakeEmbedder
    runtime=FakeRuntime()
    def new_app():return create_app(settings,runtime_factory=lambda *args:runtime,embedder_factory=FakeEmbedder,autobuild=False)
    first=new_app()
    async with first.router.lifespan_context(first):
        async with AsyncClient(transport=ASGITransport(app=first),base_url='http://testserver') as client:
            chat=await register(client)
            cookies=client.cookies
            async with first.state.db.sessions() as session:
                owner=await session.get(Chat,chat)
                run=Run(user_id=owner.user_id,chat_id=chat,request_key='stale_0001',prompt='fixture',status='running')
                session.add(run);await session.commit();run_id=run.id
    second=new_app()
    async with second.router.lifespan_context(second):
        async with AsyncClient(transport=ASGITransport(app=second),base_url='http://testserver',cookies=cookies) as client:
            data=(await client.get('/api/runs/'+run_id)).json()
            assert data['status']=='interrupted' and data['error_code']=='server_restarted'
            assert runtime.calls==0
            assert len((await client.get('/api/chats/'+chat)).json()['runs'])==1
