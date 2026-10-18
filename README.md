# vLLM на CPU в Kubernetes (Minikube)

Развёртывание OpenAI-совместимого инференса LLM на **vLLM 0.31.0** (CPU-бэкенд) в Kubernetes.
Модель по умолчанию — `Qwen/Qwen2.5-1.5B-Instruct`, меняется в одном файле.

Соответствие требованиям задания:

| Требование | Реализация | Где |
| --- | --- | --- |
| vLLM ≥ 0.27.0 на CPU | Официальные CPU-колёса vLLM 0.31.0 (`+cpu`, aarch64 и x86_64), torch 2.13.0+cpu | `docker/vllm/requirements.in` |
| Свой rootless Dockerfile | Multi-stage, пользователь `10001:10001`, в k8s — `runAsNonRoot`, `readOnlyRootFilesystem`, `drop: ALL`, seccomp `RuntimeDefault` | `docker/vllm/Dockerfile`, `k8s/deployment.yaml` |
| Полностью закреплённые зависимости | `requirements.txt` из `uv pip compile --universal`: все транзитивные зависимости с точными версиями для обеих архитектур, установка с `--no-deps`. Базовые образы и uv тоже закреплены по версии | `docker/*/requirements.txt` |
| Логи без промптов и ответов, но с фактом запроса | Фильтр в `logging.dictConfig` переписывает сообщения `RequestLogger` vLLM до вывода | `docker/vllm/vllm_logging/` |
| Кэш весов в PVC | При первом старте веса скачиваются в PVC, при рестартах читаются локально; vLLM работает с `HF_HUB_OFFLINE=1` | `k8s/pvc.yaml`, `docker/model-downloader/` |
| Secret: токен HF и Bearer-токен API | `secretGenerator` из `k8s/secrets/secrets.env` (в `.gitignore`), `VLLM_API_KEY` | `k8s/kustomization.yaml` |
| ConfigMap с параметрами vLLM | `configMapGenerator` из `k8s/config/vllm-config.yaml` → `vllm serve --config` | `k8s/config/` |
| **Бонус:** kustomize | Всё ставится одной командой `kubectl apply -k k8s/` | `k8s/kustomization.yaml` |
| **Бонус:** загрузка в initContainer отдельным лёгким rootless-образом | `python:3.12-alpine` + `huggingface_hub`, 145 МБ против 3,1 ГБ у образа vLLM | `docker/model-downloader/` |

## Структура

```text
docker/
  vllm/
    Dockerfile               # образ vLLM: builder (uv + venv) -> runtime (slim, rootless)
    requirements.in          # что ставим: vllm 0.31.0+cpu
    requirements.txt         # что ставим на самом деле: 550 строк, всё закреплено
    entrypoint.sh            # проверяет наличие весов, включает tcmalloc, запускает vllm serve
    vllm_logging/
      logging_config.json    # dictConfig для vLLM (VLLM_LOGGING_CONFIG_PATH)
      vllm_privacy_logging.py# фильтры: скрыть содержимое запросов, убрать шум /health
  model-downloader/
    Dockerfile               # лёгкий rootless-образ для initContainer
    download_model.py        # идемпотентная загрузка весов в PVC
    requirements.in/.txt
k8s/
  kustomization.yaml         # namespace, генераторы ConfigMap/Secret, теги образов
  namespace.yaml  pvc.yaml  deployment.yaml  service.yaml
  config/vllm-config.yaml    # параметры vllm serve
  config/model.env           # модель, ревизия, настройки CPU
  secrets/secrets.env.example
tests/test_privacy_logging.py
```

## Запуск в Minikube

Проверено на: macOS (Apple M5), Colima (8 CPU, 12 ГБ RAM), Minikube 1.39 (драйвер docker,
Kubernetes 1.37, containerd), kubectl 1.37. Образы мультиархитектурные по исходникам: на x86_64
сборка ставит x86_64-колёса vLLM, на arm64 — aarch64.

**Ресурсы.** Под vLLM нужно ~6–10 ГиБ RAM (веса bf16 ~3 ГБ + KV-кэш 1 ГиБ + рантайм).
Minikube запускается с 10 ГБ:

```bash
minikube start --driver=docker --cpus 7 --memory 10g --disk-size 40g
```

**1. Собрать образы и загрузить их в Minikube**

```bash
docker build -t vllm-cpu:0.31.0 docker/vllm
docker build -t vllm-model-downloader:0.1.0 docker/model-downloader
minikube image load vllm-cpu:0.31.0 vllm-model-downloader:0.1.0
```

Если Minikube использует docker runtime, можно собирать сразу внутри него: `eval $(minikube docker-env)` и те же `docker build`.

**2. Создать секреты**

```bash
cp k8s/secrets/secrets.env.example k8s/secrets/secrets.env
sed -i.bak "s/^API_TOKEN=.*/API_TOKEN=$(openssl rand -hex 32)/" k8s/secrets/secrets.env && rm k8s/secrets/secrets.env.bak
```

`HF_TOKEN` нужен только для закрытых (gated) моделей; для Qwen2.5 можно оставить пустым.

**3. Развернуть**

```bash
kubectl apply -k k8s/
kubectl -n vllm get pods -w
```

Первый старт: initContainer скачивает веса (2,9 ГБ, 1–4 минуты в зависимости от сети), затем vLLM загружает
модель (~40 секунд). Готовность определяет `startupProbe` на `/health` (до 15 минут).

**4. Отправить запрос**

```bash
kubectl -n vllm port-forward svc/vllm 8000:8000
```

```bash
TOKEN=$(grep ^API_TOKEN k8s/secrets/secrets.env | cut -d= -f2)
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-1.5b-instruct", "messages": [{"role": "user", "content": "Привет!"}], "max_tokens": 64}'
```

Без токена или с неверным токеном API отвечает `401`. `/health` открыт без токена — он нужен для проб kubelet.

## Как это работает

### Кэш весов в PVC

```text
Pod start
 ├─ initContainer model-downloader (alpine, 145 МБ, uid 10001)
 │    /models/<org>--<model>/.download-complete есть и ревизия совпадает?
 │       да  -> выход за ~1 с, без обращения к сети
 │       нет -> snapshot_download в <model>.partial -> rename -> записать маркер
 └─ container vllm (uid 10001, /models смонтирован read-only, HF_HUB_OFFLINE=1)
      entrypoint.sh: нет маркера -> ошибка; есть -> vllm serve /models/<org>--<model>
```

- Скачивание идёт во временный каталог и переименовывается только в конце: прерванная загрузка не оставит «полумодель», которую vLLM примет за готовую.
- Качаются только нужные файлы (`*.safetensors`, `*.json`, токенизатор), без дублей весов в `*.bin`/`*.gguf`.
- Ревизия модели закреплена на коммит репозитория HF (`MODEL_REVISION` в `model.env`). При смене ревизии initContainer перекачивает веса сам.
- PVC — `ReadWriteOnce`, поэтому у Deployment стратегия `Recreate`: старый под освобождает том до старта нового.

### Логирование без содержимого запросов

vLLM логирует запросы через `RequestLogger` (`--enable-log-requests`, `--enable-log-outputs`).
Вместо стандартной конфигурации подключается своя (`VLLM_LOGGING_CONFIG_PATH`) с фильтром,
который переписывает записи **до** форматирования:

| Исходное сообщение vLLM | Что попадает в лог |
| --- | --- |
| `Received request <id>: params: …, prompt: …` | `Received request <id>` |
| `Request <id> details: prompt: …, prompt_token_ids: …` (DEBUG) | `Request <id> details: [redacted]` |
| `Generated response <id>: output: '…', finish_reason: stop` | `Generated response <id>: finish_reason: stop` |
| `Generated response <id> (streaming delta): …` (на каждый токен) | не выводится |
| `Request <id> failed (bad request): <текст ошибки>` | `Request <id> failed (bad request): [redacted]` — текст ошибки может содержать ввод пользователя |
| Любое другое сообщение `vllm.*` с `prompt:` / `output:` / `JSON body:` | `Log message from <logger> suppressed: may contain request content` |
| `GET /health` от kubelet каждые 10 с | не выводится |

Фильтр работает во всех процессах vLLM (API-сервер и engine core), потому что модуль лежит в
`PYTHONPATH` образа. Пример реального лога:

```text
INFO 10-18 14:15:37 [request_logger.py:65] Received request chatcmpl-87e7c621977328e4
INFO 10-18 14:15:37 [async_llm.py:553] Added request chatcmpl-87e7c621977328e4-b8d99888.
INFO 10-18 14:15:45 [request_logger.py:109] Generated response chatcmpl-87e7c621977328e4: finish_reason: stop
INFO 10-18 14:15:45 [httptools_impl.py:482] 127.0.0.1:47268 - "POST /v1/chat/completions HTTP/1.1" 200
```

Проверка: тесты (запускаются в образе, чтобы использовать настоящий `RequestLogger` vLLM)

```bash
docker run --rm -v "$PWD/tests:/tests:ro" --entrypoint python vllm-cpu:0.31.0 -m unittest discover -s /tests -v
```

и поиск по логам пода после запросов с уникальной строкой в промпте:

```bash
kubectl -n vllm logs deploy/vllm -c vllm | grep -c "СЕКРЕТ-42"   # 0
```

### Конфигурация

- `k8s/config/vllm-config.yaml` — параметры `vllm serve` (ключи = флаги CLI без `--`): `dtype`, `max-model-len`, `max-num-seqs`, `kv-cache-memory-bytes`, логирование.
- `k8s/config/model.env` — модель, ревизия, имя модели в API, настройки CPU-бэкенда (`VLLM_CPU_OMP_THREADS_BIND`, `VLLM_CPU_NUM_OF_RESERVED_CPU`).

Изменение любого файла → `kubectl apply -k k8s/`. Kustomize добавляет к имени ConfigMap/Secret
хэш содержимого, поэтому Deployment получает новое имя и под перезапускается с новыми
значениями. Ручной `rollout restart` не нужен.

### Обновление зависимостей

```bash
cd docker/vllm
uv pip compile requirements.in --universal --python-version 3.12 --torch-backend cpu \
  --index-strategy unsafe-best-match --no-header -o requirements.txt
```

Для `docker/model-downloader` — то же без `--torch-backend`.

## Решения и проблемы, найденные при отладке

| Проблема | Причина | Решение |
| --- | --- | --- |
| Engine падал с `VLLM_PORT 'tcp://10.108.…:8000' appears to be a URI` | Service называется `vllm`, и k8s добавляет в под переменные `VLLM_PORT`, `VLLM_SERVICE_HOST`…, а vLLM читает `VLLM_*` как свои настройки | `enableServiceLinks: false` в поде |
| `InvalidCxxCompiler: No working C++ compiler found` на первом запросе | CPU-бэкенд vLLM компилирует часть ядер через torch inductor даже в eager-режиме | `g++` в runtime-образе (как в официальном CPU-образе vLLM); `enforce-eager: true` ускоряет старт |
| Первый старт падал: «Available memory 0.01/8.0 GiB … less than requested for kv» | В лимит памяти контейнера входит page cache от чтения 3 ГБ весов | Лимит 10Gi, KV-кэш 1 ГиБ (для 4 × 4096 токенов этой модели нужно ~0,5 ГиБ) |
| Шум в логах | `GET /health` каждые 10 с, строка на каждый токен стрима | Отдельные фильтры, факт запроса и итог генерации остаются |
| Нужен glibc ≥ 2.39 | CPU-колёса vLLM собраны под `manylinux_2_39` | База `python:3.12-slim-trixie` (Debian 13, glibc 2.41) |

Другие решения:

- **`readOnlyRootFilesystem`**: всё, что vLLM и torch пишут на диск (`/tmp`, `~/.cache`, `/dev/shm`), вынесено в `emptyDir`; `/dev/shm` — в памяти (2 ГиБ) для межпроцессного обмена API-сервера и engine.
- **`automountServiceAccountToken: false`**: поду не нужен доступ к API Kubernetes.
- **tcmalloc** подключается через `LD_PRELOAD` в `entrypoint.sh`, путь вычисляется по архитектуре.
- **Телеметрия отключена**: `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`, `HF_HUB_DISABLE_TELEMETRY=1`.

## Замеры (Apple M5, 7 vCPU в Minikube)

| Что | Время |
| --- | --- |
| Первый старт: скачивание весов 2,9 ГБ | 57–253 с (зависит от сети) |
| Рестарт пода до Ready (веса из PVC) | 41 с |
| Запрос: 48 токенов промпта → 64 токена ответа | ~9,3 с (~7 токенов/с); первый запрос после старта ~17 с |
| Размер образов | vLLM 3,1 ГБ, model-downloader 145 МБ |

## Удаление

```bash
kubectl delete -k k8s/        # PVC с весами удаляется вместе с namespace
minikube delete
```
