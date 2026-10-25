import pytest
from google.genai import types
from app.context import trim_contents
from app.limits import RunStopped,RunGuard
from app.tools import ToolSet

def content(role,part):return types.Content(role=role,parts=[part])

def test_whole_turns_keep_tool_pairs():
    turns=[]
    for i in range(12):
        turns.extend([content('user',types.Part(text=str(i))),content('model',types.Part(function_call=types.FunctionCall(name='count_places',args={'kind':'bank'},id='call'+str(i)))),content('user',types.Part(function_response=types.FunctionResponse(name='count_places',response={'count':1},id='call'+str(i)))),content('model',types.Part(text='done'))])
    kept=trim_contents(turns,maximum_turns=3)
    assert len(kept)==12 and kept[0].parts[0].text=='9'
    assert kept[1].parts[0].function_call.id==kept[2].parts[0].function_response.id
    with pytest.raises(RunStopped,match='context_limit'):
        trim_contents([content('user',types.Part(text='x'*100))],maximum_chars=10)

async def test_details_preserve_order_and_coordinates():
    class Poi:
        async def details(self,place_id):return {'id':place_id,'name':'fixture'}
    async def emit(*args):pass
    snapshot={'place_ids':[3,2,1],'filters':{'lat':47.23,'lon':39.7}}
    tools=ToolSet(Poi(),None,RunGuard(),emit,snapshot)
    await tools.execute('place_details',{'place_id':3})
    assert tools.last_ids==[3,2,1]
    assert snapshot['filters']['lat']==47.23

def test_usage_unknown_and_limits():
    guard=RunGuard();guard.before_model();assert guard.totals()['input_tokens'] is None
    guard.record_usage(types.GenerateContentResponseUsageMetadata(prompt_token_count=10,candidates_token_count=2))
    guard.record_usage(types.GenerateContentResponseUsageMetadata(prompt_token_count=10,candidates_token_count=2))
    assert guard.totals()['input_tokens']==10
    guard.rejected('count_places',{'kind':'x'});guard.rejected('count_places',{'kind':'x'})
    with pytest.raises(RunStopped,match='repeated_invalid_arguments'):guard.before_model()
