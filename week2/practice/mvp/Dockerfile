FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS dependencies
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=0
RUN pip install --no-cache-dir uv==0.9.5
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --python /usr/local/bin/python

FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 PATH="/app/.venv/bin:$PATH" HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false LITELLM_LOCAL_MODEL_COST_MAP=True LITELLM_TELEMETRY=False
WORKDIR /app
COPY --from=dependencies /app/.venv /app/.venv
COPY app ./app
COPY data ./data
COPY migrations ./migrations
COPY alembic.ini ./
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=40s CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2)"]
CMD ["python", "-m", "app.cli", "serve"]
