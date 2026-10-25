import asyncio
import os
from pathlib import Path
import pytest
import pytest_asyncio
from httpx import AsyncClient,ASGITransport
from app.config import Settings
from app.main import create_app

# Tests never load provider config or credentials.
os.environ['LITELLM_LOCAL_MODEL_COST_MAP']='True'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'

@pytest.fixture(autouse=True)
def external_network_blocked(monkeypatch):
    import socket
    connect,connect_ex,resolve=socket.socket.connect,socket.socket.connect_ex,socket.getaddrinfo
    def allowed(host):return host in {None,'','localhost','127.0.0.1','::1',b'localhost',b'127.0.0.1',b'::1'}
    def check(address):
        if isinstance(address,tuple) and not allowed(address[0]):raise RuntimeError('External network blocked in tests')
    def guarded_connect(sock,address):check(address);return connect(sock,address)
    def guarded_connect_ex(sock,address):check(address);return connect_ex(sock,address)
    def guarded_resolve(host,*args,**kwargs):
        if not allowed(host):raise RuntimeError('External network blocked in tests')
        return resolve(host,*args,**kwargs)
    monkeypatch.setattr(socket.socket,'connect',guarded_connect)
    monkeypatch.setattr(socket.socket,'connect_ex',guarded_connect_ex)
    monkeypatch.setattr(socket,'getaddrinfo',guarded_resolve)

class FakeEmbedder:
    def __init__(self,settings):pass
    async def encode(self,texts,query=False):return [[1.0]+[0.0]*383 for _ in texts]
    def close(self):pass

class FakeRuntime:
    def __init__(self):
        self.calls=0;self.active=0;self.peak=0;self.hold=False;self.release=asyncio.Event();self.snapshots=[]
    async def run(self,user_id,chat_id,prompt,snapshot,emit):
        self.calls+=1;self.active+=1;self.peak=max(self.peak,self.active)
        self.snapshots.append((user_id,chat_id,dict(snapshot)))
        try:
            await emit('agent_step',{'step':1})
            if self.hold:await self.release.wait()
            await asyncio.sleep(.01)
            await emit('usage',{'model_calls':1,'input_tokens':12,'output_tokens':4})
            snapshot['place_ids']=[1];snapshot['filters']={'kind':'bank','lat':47.23,'lon':39.7}
            return {'type':'count','count':42,'places':[]},dict(model_calls=1,input_tokens=12,output_tokens=4),snapshot
        finally:self.active-=1

@pytest.fixture
def settings(tmp_path):
    return Settings(database_url='sqlite+aiosqlite:///'+str(tmp_path/'test.sqlite'),
        poi_path=Path(__file__).resolve().parents[2]/'poi.sqlite',private_path=tmp_path/'private',api_key='',model='')

@pytest_asyncio.fixture
async def app(settings):
    runtime=FakeRuntime()
    app=create_app(settings,runtime_factory=lambda *args:runtime,embedder_factory=FakeEmbedder,autobuild=False)
    async with app.router.lifespan_context(app):
        app.state.fake=runtime
        yield app

@pytest_asyncio.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app),base_url='http://testserver') as client:
        yield client

async def register(client,name='tester_one'):
    response=await client.post('/api/auth/register',json={'username':name,'password':'test-only-password-2026'})
    assert response.status_code==201
    client.headers['x-csrf-token']=client.cookies['mvp_csrf']
    return (await client.post('/api/chats')).json()['id']

async def send(client,chat,key='request_0001',text='Сколько банков?'):
    response=await client.post(f'/api/chats/{chat}/messages',json={'text':text,'request_key':key})
    assert response.status_code==202,response.json()
    return response.json()['id']

async def wait_run(client,run_id):
    for _ in range(200):
        data=(await client.get('/api/runs/'+run_id)).json()
        if data['status'] in {'completed','failed','cancelled','interrupted'}:return data
        await asyncio.sleep(.01)
    raise AssertionError('run did not terminate')
