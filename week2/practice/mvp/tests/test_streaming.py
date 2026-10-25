from conftest import register,send,wait_run

async def test_sse_replay_does_not_rerun(client,app):
    chat=await register(client);run=await send(client,chat);await wait_run(client,run)
    response=await client.get('/api/runs/'+run+'/events')
    assert response.headers['content-type'].startswith('text/event-stream')
    assert 'event: agent_step' in response.text and 'event: completed' in response.text
    ids=[int(line[4:]) for line in response.text.splitlines() if line.startswith('id: ')]
    assert ids==sorted(set(ids))
    replay=await client.get('/api/runs/'+run+'/events',headers={'Last-Event-ID':str(ids[-2])})
    assert 'event: completed' in replay.text
    assert 'event: agent_step' not in replay.text
    assert app.state.fake.calls==1
    assert (await client.get('/api/runs/'+run+'/events?after=bad')).status_code==400

async def test_private_text_never_in_events(client):
    chat=await register(client);run=await send(client,chat,text='Bearer test-private-token-0000')
    data=await wait_run(client,run)
    assert 'test-private-token' not in data['prompt']
    assert 'test-private-token' not in (await client.get('/api/runs/'+run+'/events')).text
