# syntax=docker/dockerfile:1
# Streamlit demo UI. Talks to the API over HTTP only (MEDQUAD_API_URL); no torch, no models.
# Owner: service-platform-engineer. Build context: repository root.
#   docker build -f deploy/docker/ui.Dockerfile -t medquad-ui .

ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:7753c33391fc9f01d1984375bf375eb6686d52ba10db6043a86634a5ccf90dcf

FROM ${PYTHON_IMAGE} AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /build
COPY pyproject.toml requirements.lock ./
RUN grep -viE '^(torch|triton|nvidia-)' requirements.lock > /tmp/lock-no-torch.txt \
 && mkdir -p src/medquad_qa && touch src/medquad_qa/__init__.py README.md \
 && pip install -c /tmp/lock-no-torch.txt ".[ui]" \
 && pip uninstall -y medquad-qa
COPY src/ src/
RUN pip install --no-deps . && pip check

FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="medquad-ui"
RUN groupadd --system --gid 10001 medquad && useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin medquad
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/src/medquad_qa/ui/app.py /app/app.py
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_CLIENT_TOOLBAR_MODE=viewer \
    STREAMLIT_BROWSER_SERVER_ADDRESS=127.0.0.1 \
    STREAMLIT_BROWSER_SERVER_PORT=8501 \
    MEDQUAD_API_URL=http://api:8000
WORKDIR /app
USER 10001:10001
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=3s --start-period=15s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2).status == 200 else 1)"]
CMD ["streamlit", "run", "/app/app.py"]
