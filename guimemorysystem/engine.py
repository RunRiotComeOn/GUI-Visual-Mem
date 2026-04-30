"""Minimal OpenAI-compatible engine used by memory and selector agents."""
from __future__ import annotations

import logging
import time
from typing import Any, Protocol

import requests
from requests import RequestException

from guimemorysystem.images import image_to_chat_content

logger = logging.getLogger(__name__)


class EngineProtocol(Protocol):
    """Small interface required by ``GUIMemorySystem``."""

    def chat(self, messages: list[dict], max_tokens: int = 300, temperature: float | None = None) -> str:
        ...

    def chat_with_images(
        self,
        system_prompt: str,
        user_text: str,
        current_image: Any | None,
        previous_image: Any | None = None,
        max_tokens: int = 300,
        lossy: bool = True,
    ) -> str:
        ...


class OpenAICompatibleEngine:
    """Simple ``/chat/completions`` client for OpenAI-compatible services."""

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        *,
        rate_limit: int = -1,
        temperature: float = 0.0,
        timeout: int = 300,
        max_retries: int = 8,
    ) -> None:
        if not api_key:
            raise ValueError("Missing API key for OpenAICompatibleEngine.")
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.request_interval = 0.0 if rate_limit == -1 else 60.0 / max(rate_limit, 1)
        self._next_available_time = 0.0

    def _wait_for_slot(self) -> None:
        if self.request_interval <= 0:
            return
        now = time.time()
        if now < self._next_available_time:
            time.sleep(self._next_available_time - now)
        self._next_available_time = max(now, self._next_available_time) + self.request_interval

    def chat(self, messages: list[dict], max_tokens: int = 300, temperature: float | None = None) -> str:
        self._wait_for_slot()
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
        }
        last_error: Exception | None = None
        for attempt_idx in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                message = data["choices"][0]["message"]
                return message.get("content") or message.get("reasoning_content") or ""
            except RequestException as exc:
                last_error = exc
                if attempt_idx == self.max_retries - 1:
                    raise
                sleep_seconds = 2**attempt_idx
                logger.warning(
                    "OpenAI-compatible request failed on attempt %s/%s: %s. Retrying in %ss.",
                    attempt_idx + 1,
                    self.max_retries,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
        assert last_error is not None
        raise last_error

    def chat_with_images(
        self,
        system_prompt: str,
        user_text: str,
        current_image: Any | None,
        previous_image: Any | None = None,
        max_tokens: int = 300,
        lossy: bool = True,
    ) -> str:
        content: list[dict] = []
        if previous_image is not None:
            content.append(image_to_chat_content(previous_image, lossy=lossy))
        if current_image is not None:
            content.append(image_to_chat_content(current_image, lossy=lossy))
        content.append({"type": "text", "text": user_text})
        return self.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            max_tokens=max_tokens,
        )
