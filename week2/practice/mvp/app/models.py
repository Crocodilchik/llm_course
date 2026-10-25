from __future__ import annotations
from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import String, Text, DateTime, Integer, ForeignKey, JSON, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector


def uid() -> str:
    return uuid4().hex


def now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "mvp_users"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    username: Mapped[str] = mapped_column(String(32), unique=True)
    password_hash: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AuthSession(Base):
    __tablename__ = "mvp_auth_sessions"
    digest: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("mvp_users.id"), index=True)
    csrf_digest: Mapped[str] = mapped_column(String(64))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked: Mapped[bool] = mapped_column(default=False)


class Chat(Base):
    __tablename__ = "mvp_chats"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    user_id: Mapped[str] = mapped_column(ForeignKey("mvp_users.id"), index=True)
    title: Mapped[str] = mapped_column(String(100), default="Новый диалог")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    last_results: Mapped[list] = mapped_column(JSON, default=list)
    filters: Mapped[dict] = mapped_column(JSON, default=dict)


class Run(Base):
    __tablename__ = "mvp_runs"
    __table_args__ = (UniqueConstraint("chat_id", "request_key"),)
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    chat_id: Mapped[str] = mapped_column(ForeignKey("mvp_chats.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("mvp_users.id"), index=True)
    request_key: Mapped[str] = mapped_column(String(64))
    prompt: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="queued")
    final_answer: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    sequence: Mapped[int] = mapped_column(Integer, default=0)
    model_calls: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int | None] = mapped_column(Integer, default=0, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, default=0, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class RunEvent(Base):
    __tablename__ = "mvp_run_events"
    run_id: Mapped[str] = mapped_column(ForeignKey("mvp_runs.id"), primary_key=True)
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(40))
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class SearchDocument(Base):
    __tablename__ = "mvp_search_documents"
    poi_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(64), primary_key=True)
    document: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list] = mapped_column(Vector(384).with_variant(JSON(), "sqlite"))
    attributes: Mapped[dict] = mapped_column(JSON)


class Metadata(Base):
    __tablename__ = "mvp_metadata"
    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON)
