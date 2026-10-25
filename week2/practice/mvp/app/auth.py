from __future__ import annotations
import asyncio
from collections import defaultdict, deque
from datetime import timedelta, timezone
import hashlib
import hmac
import secrets
import time
from urllib.parse import urlsplit
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, InvalidHashError
from fastapi import APIRouter, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from .models import User, AuthSession, now
from .schemas import Credentials

router = APIRouter(prefix="/api/auth")
hasher = PasswordHasher(time_cost=2, memory_cost=19456, parallelism=1)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def origin_check(request: Request) -> None:
    origin = request.headers.get("origin")
    if request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(403, "cross_origin_request")
    if origin and urlsplit(origin).netloc != request.headers.get("host"):
        raise HTTPException(403, "cross_origin_request")


async def throttle(request: Request) -> None:
    bucket = digest(request.client.host if request.client else "local")
    limits = request.app.state.auth_limits
    if bucket not in limits and len(limits) >= 1000:
        raise HTTPException(429, "too_many_attempts")
    records = limits[bucket]
    stamp = time.monotonic()
    while records and stamp - records[0] > 60:
        records.popleft()
    if len(records) >= 20:
        raise HTTPException(429, "too_many_attempts")
    records.append(stamp)


async def identity(request: Request, csrf: bool = False) -> User:
    token = request.cookies.get("mvp_session", "")
    if not 20 <= len(token) <= 128:
        raise HTTPException(401, "authentication_required")
    async with request.app.state.db.sessions() as session:
        auth = await session.get(AuthSession, digest(token))
        if not auth or auth.revoked or auth.expires_at.replace(tzinfo=timezone.utc) <= now():
            raise HTTPException(401, "session_expired")
        if csrf:
            origin_check(request)
            supplied = request.headers.get("x-csrf-token", "")
            if not supplied or not hmac.compare_digest(auth.csrf_digest, digest(supplied)):
                raise HTTPException(403, "csrf_failed")
        user = await session.get(User, auth.user_id)
        if not user:
            raise HTTPException(401, "authentication_required")
        return user


async def issue_session(request: Request, response: Response, user: User) -> dict:
    token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    settings = request.app.state.settings
    seconds = settings.session_hours * 3600
    async with request.app.state.db.sessions() as session:
        session.add(AuthSession(digest=digest(token), user_id=user.id,
            csrf_digest=digest(csrf), expires_at=now() + timedelta(seconds=seconds)))
        await session.commit()
    response.set_cookie("mvp_session", token, httponly=True, secure=settings.cookie_secure,
                        samesite="lax", max_age=seconds)
    response.set_cookie("mvp_csrf", csrf, httponly=False, secure=settings.cookie_secure,
                        samesite="lax", max_age=seconds)
    return dict(username=user.username)


@router.post("/register", status_code=201)
async def register(body: Credentials, request: Request, response: Response):
    origin_check(request)
    await throttle(request)
    password_hash = await asyncio.to_thread(hasher.hash, body.password)
    user = User(username=body.username.lower(), password_hash=password_hash)
    async with request.app.state.db.sessions() as session:
        session.add(user)
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            raise HTTPException(409, "username_taken") from None
    return await issue_session(request, response, user)


@router.post("/login")
async def login(body: Credentials, request: Request, response: Response):
    origin_check(request)
    await throttle(request)
    async with request.app.state.db.sessions() as session:
        user = await session.scalar(select(User).where(User.username == body.username.lower()))
    stored = user.password_hash if user else request.app.state.dummy_password_hash
    def verify():
        try:
            return hasher.verify(stored, body.password)
        except (VerificationError, InvalidHashError):
            return False
    valid = await asyncio.to_thread(verify)
    if not valid or user is None:
        raise HTTPException(401, "invalid_credentials")
    return await issue_session(request, response, user)


@router.get("/me")
async def me(request: Request):
    user = await identity(request)
    return dict(username=user.username)


@router.post("/logout")
async def logout(request: Request, response: Response):
    await identity(request, csrf=True)
    async with request.app.state.db.sessions() as session:
        auth = await session.get(AuthSession, digest(request.cookies["mvp_session"]))
        auth.revoked = True
        await session.commit()
    response.delete_cookie("mvp_session")
    response.delete_cookie("mvp_csrf")
    return dict(ok=True)
