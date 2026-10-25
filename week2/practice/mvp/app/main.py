from __future__ import annotations
import asyncio
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import secrets
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware
from . import auth, chat, events
from .config import Settings
from .db import Database
from .agent import AgentRuntime,persistent_sessions
from .poi import PoiRepository
from .search import SearchService
from .embeddings import Embedder

ROOT=Path(__file__).resolve().parent
templates=Jinja2Templates(directory=str(ROOT/'templates'))


def create_app(settings=None, runtime_factory=None, embedder_factory=None, autobuild=True):
    @asynccontextmanager
    async def lifespan(app):
        # Provider SDKs must never log credentials, prompts or raw responses.
        logging.disable(logging.CRITICAL)
        config=settings or Settings.load()
        config.private_path.mkdir(parents=True,exist_ok=True)
        (config.private_path/'tmp').mkdir(exist_ok=True)
        app.state.settings=config
        db=Database(config.database_url)
        app.state.db=db
        await db.initialize()
        await db.recover()
        app.state.auth_limits=defaultdict(deque)
        app.state.dummy_password_hash=await asyncio.to_thread(auth.hasher.hash,secrets.token_urlsafe(32))
        poi=PoiRepository(config.poi_path)
        embedder=(embedder_factory or Embedder)(config)
        search=SearchService(db,poi,embedder)
        app.state.search=search
        sessions=None
        if runtime_factory:
            runtime=runtime_factory(config,db,poi,search)
        else:
            sessions=persistent_sessions(db.engine)
            await sessions.prepare_tables()
            runtime=AgentRuntime(config,sessions,poi,search)
        app.state.manager=chat.RunManager(db,runtime,config)
        async def index():
            try:
                await search.build()
            except Exception:
                search.state='failed'
        indexing=asyncio.create_task(index()) if autobuild else None
        try:
            yield
        finally:
            if indexing:
                indexing.cancel()
                await asyncio.gather(indexing,return_exceptions=True)
            await app.state.manager.close()
            embedder.close()
            if sessions:
                await sessions.close()
            await db.close()

    app=FastAPI(title='Рядом · Ростов',lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['localhost','127.0.0.1','testserver'])
    @app.middleware('http')
    async def headers(request,call_next):
        if request.method in {'POST','PUT','PATCH'}:
            # Bound even chunked bodies before JSON parsing; never echo the body.
            size=0;chunks=[]
            async for chunk in request.stream():
                size+=len(chunk)
                if size>32768:
                    return JSONResponse({'detail':'request_too_large'},status_code=413)
                chunks.append(chunk)
            request._body=b''.join(chunks)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='same-origin'
        response.headers['Cache-Control']='no-store'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        return response
    @app.exception_handler(RequestValidationError)
    async def validation(request,error):
        return JSONResponse({'detail':'invalid_request'},status_code=422)
    @app.exception_handler(Exception)
    async def failure(request,error):
        return JSONResponse({'detail':'internal_error'},status_code=500)
    @app.get('/health')
    async def health():
        from sqlalchemy import text
        async with app.state.db.sessions() as session:
            await session.execute(text('SELECT 1'))
        return {'status':'ok'}
    @app.get('/api/status')
    async def status(request:Request):
        await auth.identity(request)
        s=app.state.settings
        return {'search':app.state.search.state,'model_configured':bool(s.api_key and s.model and not s.api_key.startswith('<'))}
    @app.get('/')
    async def home(request:Request):
        return templates.TemplateResponse(request=request,name='auth.html')
    @app.get('/chat')
    async def page(request:Request):
        from fastapi import HTTPException
        try:
            await auth.identity(request)
        except HTTPException:
            return RedirectResponse('/',status_code=303)
        return templates.TemplateResponse(request=request,name='chat.html')
    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(events.router)
    app.mount('/static',StaticFiles(directory=str(ROOT/'static')),name='static')
    return app


app=create_app()
