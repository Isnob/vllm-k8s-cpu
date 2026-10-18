"""Скачивает веса модели с Hugging Face в примонтированный volume (PVC).

Идемпотентен: если веса уже лежат в volume (есть маркер .download-complete
с той же ревизией), выходит сразу, не обращаясь к сети. Поэтому при рестартах
пода модель грузится локально.

Переменные окружения:
  MODEL_ID        id модели на Hugging Face, например Qwen/Qwen2.5-1.5B-Instruct (обязательно)
  MODEL_REVISION  ветка / тег / коммит (по умолчанию main)
  MODELS_DIR      каталог volume (по умолчанию /models)
  HF_TOKEN        токен Hugging Face для закрытых (gated) моделей (необязательно)
"""

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

# Только то, что нужно vLLM для инференса: без дублирующих форматов весов (*.bin, *.pt, *.gguf, onnx).
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.py", "*.jinja"]
MARKER = ".download-complete"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # без построчного лога каждого HTTP-запроса
log = logging.getLogger("model-downloader")


def model_dir(models_dir: Path, model_id: str) -> Path:
    # Должно совпадать с entrypoint.sh основного контейнера.
    return models_dir / model_id.replace("/", "--")


def main() -> int:
    model_id = os.environ.get("MODEL_ID", "").strip()
    if not model_id:
        log.error("MODEL_ID не задан")
        return 1
    revision = os.environ.get("MODEL_REVISION", "main").strip() or "main"
    models_dir = Path(os.environ.get("MODELS_DIR", "/models"))
    target = model_dir(models_dir, model_id)
    marker = target / MARKER

    if marker.is_file():
        meta = json.loads(marker.read_text())
        if meta.get("model_id") == model_id and meta.get("revision") == revision:
            log.info("Веса %s@%s уже в %s — загрузка не нужна", model_id, revision, target)
            return 0
        log.info("В %s лежит другая ревизия (%s) — перекачиваю", target, meta.get("revision"))

    # Качаем во временный каталог рядом и переименовываем в конце: прерванная загрузка
    # не оставит «полу-модель», которую vLLM примет за готовую.
    tmp = target.with_name(target.name + ".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    log.info("Скачиваю %s@%s в %s", model_id, revision, target)
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=tmp,
        allow_patterns=ALLOW_PATTERNS,
        token=os.environ.get("HF_TOKEN") or None,
    )
    shutil.rmtree(tmp / ".cache", ignore_errors=True)  # служебный кэш huggingface_hub не нужен vLLM
    if target.exists():
        shutil.rmtree(target)
    tmp.rename(target)
    marker.write_text(json.dumps({"model_id": model_id, "revision": revision}))

    size_gb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1024**3
    log.info("Готово: %.2f ГБ за %.0f с", size_gb, time.monotonic() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
