"""Application-only configuration. Values must never be logged or serialized."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from sqlalchemy.engine import URL


@dataclass(repr=False)
class Settings:
    database_url: str
    poi_path: Path
    private_path: Path
    api_key: str = field(default="", repr=False)
    model: str = ""
    api_base: str = "https://openrouter.ai/api/v1"
    active_runs: int = 3
    run_timeout: int = 90
    session_hours: int = 24
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_revision: str | None = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    model_cache: Path = Path(__file__).resolve().parents[2] / ".cache/models"
    cookie_secure: bool = False

    @classmethod
    def load(cls) -> "Settings":
        from dotenv import load_dotenv
        here = Path(__file__).resolve().parents[1]
        # The application may consume config indirectly; never expose its values.
        candidates=[here/".env",here.parent/".env"]
        if len(here.parents)>2:
            candidates.append(here.parents[2]/".env")
        for file in candidates:
            if file.is_file():
                load_dotenv(file, override=False)
                break
        private = Path(os.getenv("MVP_PRIVATE_PATH", str(here.parent / "private/mvp/app"))).resolve()
        if os.getenv("MVP_DATABASE_URL"):
            url = os.getenv("MVP_DATABASE_URL", "")
        else:
            password_path = Path(os.getenv("MVP_DB_PASSWORD_FILE", str(private / "db_password")))
            password = password_path.read_text(encoding="utf-8").strip()
            url = URL.create("postgresql+asyncpg", username="mvp", password=password,
                             host=os.getenv("MVP_DB_HOST", "127.0.0.1"), database="mvp").render_as_string(hide_password=False)
        return cls(database_url=url,
            poi_path=Path(os.getenv("MVP_POI_PATH", str(here.parent / "poi.sqlite"))).resolve(),
            private_path=private, api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("MODEL", ""), api_base=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
            active_runs=max(1, min(5, int(os.getenv("MVP_ACTIVE_RUNS", "3")))),
            run_timeout=max(15, min(180, int(os.getenv("MVP_RUN_TIMEOUT", "90")))),
            session_hours=max(1, min(168, int(os.getenv("MVP_SESSION_HOURS", "24")))),
            embedding_model=os.getenv("MVP_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
            embedding_revision=os.getenv("MVP_EMBEDDING_REVISION", "614241f622f53c4eeff9890bdc4f31cfecc418b3"),
            model_cache=Path(os.getenv("MVP_MODEL_CACHE", str(here.parent / ".cache/models"))).resolve(),
            cookie_secure=os.getenv("MVP_COOKIE_SECURE", "0") == "1")
