from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import os


@dataclass(frozen=True)
class LlmConfig:
    model: str = "qwen3:8b"
    temperature: float = 0.7
    num_predict: int = 450
    timeout_s: float = 90.0
    keep_alive: str = "10m"


class LlmAgent:
    """
    Thin wrapper around an LLM chat call with shared safety constraints.

    This project intentionally avoids "micro-action manipulation scripts" and instead
    generates consent-first, low-pressure, honest communication plans.
    """

    def __init__(self, *, llm: Optional[LlmConfig] = None):
        self.llm = llm or LlmConfig()
        self._client = None

    def _get_client(self):
        try:
            import ollama  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: the Python package 'ollama' is not installed.\n"
                "Install it with: pip install ollama\n"
                "Then make sure the Ollama app/service is running and the model is pulled "
                f"(e.g. 'ollama pull {self.llm.model}')."
            ) from e

        if self._client is None:
            # ollama.Client passes kwargs to the underlying HTTP client (httpx),
            # so we can set a hard timeout to avoid indefinite hangs.
            timeout = float(os.getenv("SITUAITION_TIMEOUT_S", str(self.llm.timeout_s)))
            self._client = ollama.Client(timeout=timeout)
        return self._client

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        client = self._get_client()
        resp: dict[str, Any] = client.chat(
            model=self.llm.model,
            messages=[{"role": "user", "content": prompt}],
            format="json" if json_mode else None,
            options={
                "temperature": self.llm.temperature,
                "num_predict": self.llm.num_predict,
            },
            keep_alive=self.llm.keep_alive,
        )
        return str(resp.get("message", {}).get("content", "")).strip()

