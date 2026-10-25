import json
import pytest
from pydantic import PrivateAttr
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.genai import types
from app.agent import AgentRuntime,AgentFailure
from app.poi import PoiRepository

class ScriptedModel(BaseLlm):
    model:str='fixture'
    _responses:list=PrivateAttr(default_factory=list)
    _calls:int=PrivateAttr(default=0)
    def __init__(self,responses):
        super().__init__();self._responses=responses
    async def generate_content_async(self,llm_request,stream=False):
        index=min(self._calls,len(self._responses)-1);self._calls+=1
        value=self._responses[index]
        part=types.Part(function_call=types.FunctionCall(name=value[0],args=value[1],id='fixture'+str(self._calls))) if isinstance(value,tuple) else types.Part(text=json.dumps(value))
        yield LlmResponse(content=types.Content(role='model',parts=[part]),usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=10,candidates_token_count=2))

async def test_adk_real_tool_roundtrip(settings):
    poi=PoiRepository(settings.poi_path);actual=(await poi.count('bank'))['count']
    model=ScriptedModel([('count_places',{'kind':'bank'}),{'type':'count','count':actual}])
    runtime=AgentRuntime(settings,InMemorySessionService(),poi,None,model_factory=lambda:model)
    events=[]
    async def emit(kind,payload):events.append((kind,payload))
    result,usage,state=await runtime.run('u','c','Сколько банков?',{},emit)
    assert result['count']==actual and usage['model_calls']==2
    assert usage['input_tokens']==20
    assert any(k=='tool_finished' for k,p in events)

async def test_bad_arguments_stop_early(settings):
    model=ScriptedModel([('count_places',{'kind':'bank','unknown':True})])
    runtime=AgentRuntime(settings,InMemorySessionService(),PoiRepository(settings.poi_path),None,lambda:model)
    async def emit(*args):pass
    with pytest.raises(AgentFailure) as failure:await runtime.run('u','c','test',{},emit)
    assert failure.value.code=='repeated_invalid_arguments'
    assert failure.value.totals['model_calls']==2

async def test_unverified_answer_rejected(settings):
    model=ScriptedModel([{'type':'count','count':999999}])
    runtime=AgentRuntime(settings,InMemorySessionService(),PoiRepository(settings.poi_path),None,lambda:model)
    async def emit(*args):pass
    with pytest.raises(AgentFailure,match='invalid_answer'):
        await runtime.run('u','c','test',{},emit)
    assert model._calls==3

async def test_unknown_tool_loop_stops_early(settings):
    model=ScriptedModel([('tool_that_does_not_exist',{})])
    runtime=AgentRuntime(settings,InMemorySessionService(),PoiRepository(settings.poi_path),None,lambda:model)
    async def emit(*args):pass
    with pytest.raises(AgentFailure) as failure:await runtime.run('u','c','test',{},emit)
    assert failure.value.code=='repeated_invalid_arguments'
    assert model._calls==2

async def test_valid_tool_loop_has_six_call_limit(settings):
    model=ScriptedModel([('count_places',{'kind':'bank'})])
    runtime=AgentRuntime(settings,InMemorySessionService(),PoiRepository(settings.poi_path),None,lambda:model)
    async def emit(*args):pass
    with pytest.raises(AgentFailure) as failure:await runtime.run('u','c','test',{},emit)
    assert failure.value.code=='model_call_limit' and model._calls==6

async def test_model_error_message_never_persisted(settings,tmp_path):
    from app.db import Database
    from app.agent import persistent_sessions
    class Broken(ScriptedModel):
        async def generate_content_async(self,*args,**kwargs):
            raise RuntimeError('private-fixture-error-value')
            yield
    db=Database(settings.database_url);await db.initialize()
    sessions=persistent_sessions(db.engine);await sessions.prepare_tables()
    runtime=AgentRuntime(settings,sessions,PoiRepository(settings.poi_path),None,lambda:Broken([]))
    async def emit(*args):pass
    try:
        with pytest.raises(AgentFailure):await runtime.run('u','c','test',{},emit)
        saved=await sessions.get_session(app_name='rostov_mvp',user_id='u',session_id='c')
        assert 'private-fixture-error-value' not in saved.model_dump_json()
    finally:await sessions.close();await db.close()

async def test_source_failure_stops_after_two(settings):
    class Broken:
        async def count(self,**args):raise OSError('fixture outage')
    model=ScriptedModel([('count_places',{'kind':'bank'})])
    runtime=AgentRuntime(settings,InMemorySessionService(),Broken(),None,lambda:model)
    async def emit(*args):pass
    with pytest.raises(AgentFailure) as failure:await runtime.run('u','c','test',{},emit)
    assert failure.value.code=='repeated_source_failure' and model._calls==2
