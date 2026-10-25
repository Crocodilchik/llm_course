from datetime import timedelta
from sqlalchemy import select
from app.models import User,AuthSession,now
from conftest import register

async def test_registration_hash_cookie_logout(client,app):
    await register(client)
    assert (await client.get('/api/auth/me')).json()['username']=='tester_one'
    async with app.state.db.sessions() as session:
        user=await session.scalar(select(User))
        assert user.password_hash.startswith('$argon2id$')
        auth=await session.scalar(select(AuthSession))
        assert auth.digest!=client.cookies['mvp_session']
    old=client.cookies['mvp_session']
    assert (await client.post('/api/auth/logout')).status_code==200
    client.cookies.set('mvp_session',old)
    assert (await client.get('/api/auth/me')).status_code==401

async def test_csrf_origin_validation_and_login(client):
    bad=await client.post('/api/auth/register',json={'username':'a','password':'private-test-value'})
    assert bad.status_code==422
    assert 'private-test-value' not in bad.text
    await register(client)
    assert (await client.post('/api/chats',headers={'x-csrf-token':'wrong'})).status_code==403
    assert (await client.post('/api/chats',headers={'origin':'https://example.invalid'})).status_code==403
    assert (await client.post('/api/auth/login',json={'username':'tester_one','password':'wrong-test-password'})).status_code==401
    assert (await client.post('/api/auth/login',json={'username':'tester_one','password':'test-only-password-2026'})).status_code==200

async def test_expired_session(client,app):
    await register(client)
    async with app.state.db.sessions() as session:
        auth=await session.scalar(select(AuthSession));auth.expires_at=now()-timedelta(seconds=1);await session.commit()
    assert (await client.get('/api/chats')).status_code==401

async def test_auth_throttled(client):
    for _ in range(20):
        assert (await client.post('/api/auth/login',json={'username':'unknown','password':'test-only-password-2026'})).status_code==401
    assert (await client.post('/api/auth/login',json={'username':'unknown','password':'test-only-password-2026'})).status_code==429

async def test_oversized_body_rejected_without_echo(client):
    response=await client.post('/api/auth/register',content='x'*33000)
    assert response.status_code==413 and response.json()=={'detail':'request_too_large'}
