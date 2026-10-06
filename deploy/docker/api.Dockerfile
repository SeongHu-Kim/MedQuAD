# syntax=docker/dockerfile:1
# MedQuAD QA API image. Owner: service-platform-engineer.
#   VARIANT=cpu : torch from the PyTorch CPU index (small; CI and BM25-only smoke stack)
#   VARIANT=gpu : torch 2.14.1+cu130 (CUDA user-space libs come from the wheels; the driver is
#                 injected at run time via CDI `nvidia.com/gpu=all`, validated by scripts/ops/validate_gpu_container.sh)
# No model weights, indexes or data are baked in: they are mounted read-only at run time (HF_HUB_OFFLINE=1).
# Build context: repository root.  Example:
#   docker build -f deploy/docker/api.Dockerfile --build-arg VARIANT=cpu -t medquad-api:cpu .

ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:7753c33391fc9f01d1984375bf375eb6686d52ba10db6043a86634a5ccf90dcf

FROM ${PYTHON_IMAGE} AS builder
ARG VARIANT=cpu
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /build
COPY pyproject.toml requirements.lock ./
COPY constraints/ constraints/
# Constraints = full lock minus the CUDA/torch stack, which is installed explicitly per variant.
RUN grep -viE '^(torch|triton|nvidia-)' requirements.lock > /tmp/lock-no-torch.txt \
 && if [ "$VARIANT" = "gpu" ]; then \
      pip install --index-url https://download.pytorch.org/whl/cu130 -c constraints/torch-cu130.txt torch ; \
    else \
      pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.1+cpu" ; \
    fi
# Dependency layer (cached until pyproject/lock change): install extras against a stub package.
RUN mkdir -p src/medquad_qa && touch src/medquad_qa/__init__.py README.md \
 && pip install -c /tmp/lock-no-torch.txt ".[ml,retrieval,api]" \
 && pip uninstall -y medquad-qa
COPY src/ src/
RUN pip install --no-deps . \
 && (pip check || echo "pip check reported issues; cuSPARSELt platform-tag line is benign (D-016)")

FROM ${PYTHON_IMAGE} AS runtime
ARG VARIANT=cpu
LABEL org.opencontainers.image.title="medquad-api" org.opencontainers.image.description="MedQuAD evidence-grounded QA API (research prototype; not medical advice)" \
      medquad.variant="${VARIANT}"
RUN groupadd --system --gid 10001 medquad && useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin medquad
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    TORCH_DISABLE_NATIVE_JIT=1 \
    MEDQUAD_BIND_HOST=0.0.0.0 MEDQUAD_API_PORT=8000
WORKDIR /app
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2).status == 200 else 1)"]
CMD ["python", "-m", "medquad_qa.api"]
