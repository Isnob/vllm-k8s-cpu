"""Проверка, что фильтр логов не пропускает промпты и ответы модели.

Запуск в образе vLLM (из корня репозитория):
  docker run --rm -v "$PWD/tests:/tests:ro" --entrypoint python vllm-cpu:0.31.0 -m unittest discover -s /tests -v
"""

import io
import json
import logging
import logging.config
import unittest
from pathlib import Path

from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.sampling_params import SamplingParams

SECRET_PROMPT = "мой пароль от банка 12345"
SECRET_OUTPUT = "ответ модели с личными данными"
CONFIG_PATH = Path("/opt/vllm_logging/logging_config.json")


class PrivacyLoggingTest(unittest.TestCase):
    def setUp(self) -> None:
        config = json.loads(CONFIG_PATH.read_text())
        config["handlers"]["vllm"]["stream"] = "ext://sys.stdout"
        logging.config.dictConfig(config)
        self.buffer = io.StringIO()
        # Логгеры vllm и uvicorn.access пишут в один и тот же обработчик.
        handler = logging.getLogger("vllm").handlers[0]
        handler.setStream(self.buffer)

    def _logs(self) -> str:
        return self.buffer.getvalue()

    def test_request_fact_logged_without_prompt(self) -> None:
        RequestLogger(max_log_len=None).log_inputs(
            "chatcmpl-1", SECRET_PROMPT, [1, 2, 3], None, SamplingParams(max_tokens=16), None
        )
        logs = self._logs()
        self.assertIn("Received request chatcmpl-1", logs)
        self.assertNotIn(SECRET_PROMPT, logs)

    def test_output_not_logged_but_completion_is(self) -> None:
        RequestLogger(max_log_len=None).log_outputs("chatcmpl-2", SECRET_OUTPUT, [4, 5], finish_reason="stop")
        logs = self._logs()
        self.assertIn("Generated response chatcmpl-2: finish_reason: stop", logs)
        self.assertNotIn(SECRET_OUTPUT, logs)

    def test_debug_level_still_redacted(self) -> None:
        logging.getLogger("vllm").setLevel(logging.DEBUG)
        logging.getLogger("vllm").handlers[0].setLevel(logging.DEBUG)
        request_logger = RequestLogger(max_log_len=None)
        request_logger.log_inputs("chatcmpl-3", SECRET_PROMPT, [1, 2, 3], None, SamplingParams(), None)
        request_logger.log_outputs("chatcmpl-3", SECRET_OUTPUT, [4, 5], finish_reason="length", is_streaming=True)
        logs = self._logs()
        self.assertNotIn(SECRET_PROMPT, logs)
        self.assertNotIn(SECRET_OUTPUT, logs)
        self.assertNotIn("[1, 2, 3]", logs)
        self.assertIn("Request chatcmpl-3 details: [redacted]", logs)

    def test_bad_request_error_text_hidden(self) -> None:
        logging.getLogger("vllm.v1.engine.async_llm").info(
            "Request %s failed (bad request): %s.", "chatcmpl-4", f"invalid input: {SECRET_PROMPT}"
        )
        logs = self._logs()
        self.assertIn("Request chatcmpl-4 failed (bad request): [redacted]", logs)
        self.assertNotIn(SECRET_PROMPT, logs)

    def test_unknown_message_with_prompt_suppressed(self) -> None:
        logging.getLogger("vllm.some.new.module").info("prompt: %s", SECRET_PROMPT)
        logs = self._logs()
        self.assertNotIn(SECRET_PROMPT, logs)
        self.assertIn("suppressed", logs)

    def test_regular_logs_untouched(self) -> None:
        logging.getLogger("vllm.engine").info("Avg prompt throughput: %.1f tokens/s", 12.5)
        self.assertIn("Avg prompt throughput: 12.5 tokens/s", self._logs())

    def test_streaming_deltas_dropped_completion_kept(self) -> None:
        request_logger = RequestLogger(max_log_len=None)
        for token in ("При", "вет"):
            request_logger.log_outputs("chatcmpl-5", token, [1], finish_reason=None, is_streaming=True, delta=True)
        request_logger.log_outputs("chatcmpl-5", SECRET_OUTPUT, [1, 2], finish_reason="stop", is_streaming=True)
        logs = self._logs()
        self.assertNotIn("streaming delta", logs)
        self.assertIn("Generated response chatcmpl-5 (streaming complete): finish_reason: stop", logs)
        self.assertNotIn(SECRET_OUTPUT, logs)

    def test_health_checks_dropped_api_calls_kept(self) -> None:
        access = logging.getLogger("uvicorn.access")
        access.info('%s - "%s %s HTTP/%s" %d', "10.244.0.1:51804", "GET", "/health", "1.1", 200)
        access.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:47268", "POST", "/v1/chat/completions", "1.1", 200)
        logs = self._logs()
        self.assertNotIn("/health", logs)
        self.assertIn("POST /v1/chat/completions", logs)


if __name__ == "__main__":
    unittest.main()
