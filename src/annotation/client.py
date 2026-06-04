"""OpenRouter chat completions client (stdlib HTTP)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from annotation.schema import batch_response_format

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-opus-4.8"


class OpenRouterError(RuntimeError):
    pass


def chat_completion(
    *,
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout_s: float = 180.0,
    response_format: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": response_format or batch_response_format(),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get(
            "OPENROUTER_HTTP_REFERER", "https://github.com/bias-subspaces"
        ),
        "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "bias-subspaces-annotation"),
    }
    if extra_headers:
        headers.update(extra_headers)

    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise OpenRouterError(f"HTTP {e.code}: {err_body}") from e
    except urllib.error.URLError as e:
        raise OpenRouterError(str(e)) from e

    choices = payload.get("choices") or []
    if not choices:
        raise OpenRouterError(f"empty choices in response: {payload!r}")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        raise OpenRouterError(f"empty message content: {payload!r}")

    return {
        "content": content,
        "usage": payload.get("usage") or {},
        "model": payload.get("model", model),
        "id": payload.get("id"),
    }


def chat_with_retries(
    *,
    max_retries: int = 4,
    retry_sleep_s: float = 2.0,
    **kwargs: Any,
) -> dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            return chat_completion(**kwargs)
        except OpenRouterError as e:
            last_err = e
            msg = str(e).lower()
            retryable = any(
                x in msg
                for x in ("429", "500", "502", "503", "504", "timeout", "timed out")
            )
            if not retryable or attempt == max_retries - 1:
                raise
            time.sleep(retry_sleep_s * (2**attempt))
    raise last_err  # pragma: no cover
