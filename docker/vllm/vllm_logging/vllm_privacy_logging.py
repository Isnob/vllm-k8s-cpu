"""Фильтр логирования vLLM: факт обращения к модели логируется, содержимое — нет.

Подключается через VLLM_LOGGING_CONFIG_PATH (см. logging_config.json) и работает
во всех процессах vLLM (API-сервер и engine core), т.к. модуль лежит в PYTHONPATH.

Что делает фильтр:
- RequestLogger ("vllm.entrypoints.serve.utils.request_logger"):
    * "Received request <id>: params: ..."        -> "Received request <id>"
    * "Request <id> details: prompt: ..."         -> "Request <id> details: [redacted]"
    * "Request <id> JSON body: ..."               -> "Request <id> JSON body: [redacted]"
    * "Generated response <id>: output: ..., finish_reason: X"
                                                  -> "Generated response <id>: finish_reason: X"
    * "Generated response <id> details: output_token_ids: ..."
                                                  -> "Generated response <id> details: [redacted]"
    * "Generated response <id> (streaming delta): ..." — по строке на каждый токен стрима,
      отбрасывается; остаётся итоговая "Generated response <id> (streaming complete): ...".
- Сообщения об ошибках запроса ("Request <id> failed (bad request): <err>") — текст
  ошибки может содержать фрагменты пользовательского ввода, поэтому он скрывается.
- Страховка: если в итоговом сообщении любого логгера vllm встречаются маркеры
  содержимого запроса (prompt:, output:, JSON body:), сообщение заменяется целиком.
"""

import logging
import re

REDACTED = "[redacted]"
REQUEST_LOGGER = "vllm.entrypoints.serve.utils.request_logger"

_FAILED_WITH_DETAILS = re.compile(r"^Request %s failed \((bad request|input error)\): %s")
_CONTENT_MARKERS = re.compile(r"\b(prompt|prompt_token_ids|output|output_token_ids|JSON body)\s*:", re.IGNORECASE)


def _arg(args: tuple, index: int, default: object = "N/A") -> object:
    return args[index] if len(args) > index else default


def _is_streaming_delta(msg: str, args: tuple) -> bool:
    return msg.startswith("Generated response") and "streaming delta" in str(_arg(args, 1, ""))


class PromptRedactionFilter(logging.Filter):
    """Переписывает записи лога так, чтобы в них не было промптов и ответов модели."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not isinstance(record.msg, str):
            return True
        args = record.args if isinstance(record.args, tuple) else ()

        if record.name == REQUEST_LOGGER:
            if _is_streaming_delta(record.msg, args):
                return False  # строка на каждый токен стрима; итог пишется строкой "streaming complete"
            self._redact_request_logger(record, args)
        elif failed := _FAILED_WITH_DETAILS.match(record.msg):
            record.msg = "Request %s failed (" + failed.group(1) + "): " + REDACTED
            record.args = (_arg(args, 0),)

        # Страховка от новых или неучтённых сообщений с содержимым запроса.
        if record.name.startswith("vllm") and _CONTENT_MARKERS.search(record.getMessage()):
            record.msg = "Log message from %s suppressed: may contain request content"
            record.args = (record.name,)
        return True

    @staticmethod
    def _redact_request_logger(record: logging.LogRecord, args: tuple) -> None:
        msg = record.msg
        if msg.startswith("Received request"):
            record.msg, record.args = "Received request %s", (_arg(args, 0),)
        elif msg.startswith("Request %s details") or msg.startswith("Request %s JSON body"):
            label = "details" if "details" in msg else "JSON body"
            record.msg, record.args = "Request %s " + label + ": " + REDACTED, (_arg(args, 0),)
        elif msg.startswith("Generated response") and "finish_reason" in msg:
            record.msg = "Generated response %s%s: finish_reason: %s"
            record.args = (_arg(args, 0), _arg(args, 1, ""), _arg(args, len(args) - 1, None))
        elif msg.startswith("Generated response"):
            record.msg = "Generated response %s%s details: " + REDACTED
            record.args = (_arg(args, 0), _arg(args, 1, ""))


class HealthCheckAccessFilter(logging.Filter):
    """Убирает из access-лога проверки /health от kubelet (каждые 10 секунд, шум без пользы).

    Обращения к API (POST /v1/chat/completions и т. п.) остаются в логе: метод, путь, код ответа.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args if isinstance(record.args, tuple) else ()
        return not (record.name == "uvicorn.access" and len(args) >= 3 and args[2] == "/health")
