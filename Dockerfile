FROM python:3.11-slim

# Install system deps for Tree-sitter native builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl patch \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir uvicorn[standard]

COPY src/ ./src/
COPY scripts/ ./scripts/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "-m", "src.api.server"]
