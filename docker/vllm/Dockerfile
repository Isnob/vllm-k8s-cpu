# syntax=docker/dockerfile:1.7
# Rootless-образ vLLM для инференса на CPU (linux/arm64 и linux/amd64).

# ---------- Стадия 1: установка закреплённых зависимостей в venv ----------
FROM python:3.12-slim-trixie AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /build
COPY requirements.txt .

# --no-deps: ставим ровно то, что закреплено в requirements.txt, без самостоятельного разрешения зависимостей.
RUN uv venv /opt/venv \
 && uv pip install --python /opt/venv/bin/python --no-deps --no-cache \
      --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt

# ---------- Стадия 2: итоговый образ ----------
FROM python:3.12-slim-trixie

# tcmalloc и numa нужны CPU-бэкенду vLLM; libgomp — OpenMP для torch.
# g++ нужен torch inductor: часть CPU-ядер vLLM компилируется при первом запросе даже с enforce-eager.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libtcmalloc-minimal4t64 libnuma1 libgomp1 g++ \
 && rm -rf /var/lib/apt/lists/*

# Непривилегированный пользователь с фиксированным UID/GID (совпадает с securityContext в k8s).
RUN groupadd --gid 10001 vllm \
 && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/vllm --shell /usr/sbin/nologin vllm \
 && mkdir -p /models \
 && chown 10001:10001 /models

COPY --from=builder /opt/venv /opt/venv
COPY --chown=root:root vllm_logging/ /opt/vllm_logging/
COPY --chown=root:root entrypoint.sh /usr/local/bin/entrypoint.sh

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/vllm_logging \
    PYTHONUNBUFFERED=1 \
    VLLM_CONFIGURE_LOGGING=1 \
    VLLM_LOGGING_CONFIG_PATH=/opt/vllm_logging/logging_config.json \
    HOME=/home/vllm \
    HF_HOME=/home/vllm/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    MODELS_DIR=/models

USER 10001:10001
WORKDIR /home/vllm
EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
