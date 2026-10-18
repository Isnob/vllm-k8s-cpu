#!/usr/bin/env bash
# Запуск vLLM с локальными весами из примонтированного volume.
# Веса скачивает initContainer (см. docker/model-downloader); здесь сеть к Hugging Face не нужна.
set -euo pipefail

: "${MODEL_ID:?MODEL_ID не задан (ConfigMap vllm-model)}"
MODELS_DIR="${MODELS_DIR:-/models}"
SERVE_CONFIG_FILE="${SERVE_CONFIG_FILE:-/etc/vllm/config.yaml}"

# Путь должен совпадать с тем, куда кладёт веса download_model.py: "Qwen/Qwen2.5-1.5B-Instruct" -> "Qwen--Qwen2.5-1.5B-Instruct".
MODEL_PATH="${MODELS_DIR}/${MODEL_ID//\//--}"
if [[ ! -f "${MODEL_PATH}/.download-complete" ]]; then
  echo "Веса модели ${MODEL_ID} не найдены в ${MODEL_PATH}. Их должен скачать initContainer model-downloader." >&2
  exit 1
fi

# tcmalloc ускоряет CPU-бэкенд vLLM; путь зависит от архитектуры (aarch64 / x86_64).
TCMALLOC="/usr/lib/$(uname -m)-linux-gnu/libtcmalloc_minimal.so.4"
if [[ -f "${TCMALLOC}" ]]; then
  export LD_PRELOAD="${TCMALLOC}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

echo "Запуск vLLM: модель ${MODEL_ID} из ${MODEL_PATH}, параметры из ${SERVE_CONFIG_FILE}"
exec vllm serve "${MODEL_PATH}" \
  --config "${SERVE_CONFIG_FILE}" \
  --served-model-name "${SERVED_MODEL_NAME:-${MODEL_ID}}" \
  "$@"
