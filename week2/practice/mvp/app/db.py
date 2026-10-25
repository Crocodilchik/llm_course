from __future__ import annotations
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from .models import Base, Run


class Database:
    def __init__(self, url: str):
        self.engine = create_async_engine(url, echo=False, pool_pre_ping=True)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def initialize(self) -> None:
        async with self.engine.begin() as connection:
            if self.engine.dialect.name == "postgresql":
                await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            def migrate(sync_connection):
                from pathlib import Path
                from alembic.config import Config
                from alembic import command
                config=Config(str(Path(__file__).resolve().parents[1]/'alembic.ini'))
                config.attributes['connection']=sync_connection
                command.upgrade(config,'head')
            await connection.run_sync(migrate)

    async def recover(self) -> None:
        from sqlalchemy import update
        async with self.sessions() as session:
            await session.execute(update(Run).where(Run.status.in_(["queued", "running"])).values(
                status="interrupted", error_code="server_restarted"))
            await session.commit()

    async def close(self) -> None:
        await self.engine.dispose()
