from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class OpenAISettings:
    """Runtime settings for optional OpenAI-powered features.

    The API key is read from an explicit override first and then from
    ``OPENAI_API_KEY``. It is never written to disk by this module.
    """

    api_key: str | None
    model: str = "gpt-5.6"
    timeout_seconds: float = 45.0
    max_retries: int = 2

    @property
    def configured(self) -> bool:
        return bool((self.api_key or "").strip())

    @property
    def masked_key(self) -> str:
        key = (self.api_key or "").strip()
        if not key:
            return "Not configured"
        if len(key) <= 8:
            return "Configured"
        return f"{key[:4]}…{key[-4:]}"


def get_openai_settings(
    *,
    api_key_override: str | None = None,
    model_override: str | None = None,
) -> OpenAISettings:
    api_key = (api_key_override or os.getenv("OPENAI_API_KEY") or "").strip() or None
    model = (model_override or os.getenv("OPENAI_MODEL") or "gpt-5.6").strip()
    try:
        timeout_seconds = max(5.0, float(os.getenv("OPENAI_TIMEOUT_SECONDS", "45")))
    except ValueError:
        timeout_seconds = 45.0
    try:
        max_retries = max(0, int(os.getenv("OPENAI_MAX_RETRIES", "2")))
    except ValueError:
        max_retries = 2
    return OpenAISettings(
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
