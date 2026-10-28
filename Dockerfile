# One image, two services (mcp | agents) selected by CMD. Python 3.12 slim, non-root.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN useradd -m -u 10001 app
COPY pyproject.toml README.md ./
COPY src ./src
COPY agents ./agents
COPY data ./data
RUN pip install --no-cache-dir -e ".[pii]" \
 && python -m spacy download en_core_web_lg >/dev/null 2>&1 || true
USER app
EXPOSE 8765 8080
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request,os;urllib.request.urlopen(os.environ.get('HEALTH_URL','http://localhost:8765/healthz'))" || exit 1
CMD ["python", "-m", "store_support.mcp_server.server"]
