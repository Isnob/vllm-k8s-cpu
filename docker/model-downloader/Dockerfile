# Лёгкий rootless-образ для initContainer: только Python и huggingface_hub, без torch и vLLM.
FROM python:3.12-alpine

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-deps --only-binary=:all: -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt \
 && addgroup -g 10001 downloader \
 && adduser -u 10001 -G downloader -D -h /home/downloader -s /sbin/nologin downloader

COPY --chown=root:root download_model.py /app/download_model.py

ENV PYTHONUNBUFFERED=1 \
    HOME=/home/downloader \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MODELS_DIR=/models

USER 10001:10001
ENTRYPOINT ["python", "/app/download_model.py"]
