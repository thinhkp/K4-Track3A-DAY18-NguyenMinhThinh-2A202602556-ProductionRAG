"""Select OpenAI or Groq without changing the retrieval embedding model."""

import threading
import time

import config

_rate_lock = threading.Lock()
_next_request_at = 0.0


def llm_settings() -> dict:
    provider = config.LLM_PROVIDER
    if provider == "auto":
        provider = "groq" if config.GROQ_API_KEY else "openai"
    if provider == "groq":
        return {"provider": provider, "api_key": config.GROQ_API_KEY,
                "model": config.GROQ_MODEL, "base_url": config.GROQ_BASE_URL}
    if provider == "openai":
        return {"provider": provider, "api_key": config.OPENAI_API_KEY,
                "model": config.OPENAI_MODEL, "base_url": None}
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def chat_client():
    """Return the OpenAI-compatible client and configured model."""
    from openai import OpenAI

    settings = llm_settings()
    if not settings["api_key"]:
        raise RuntimeError(f"{settings['provider'].upper()}_API_KEY is missing")
    options = {"api_key": settings["api_key"]}
    if settings["base_url"]:
        options["base_url"] = settings["base_url"]
    return OpenAI(**options), settings["model"]


def chat_completion(**kwargs):
    """Call the configured provider while respecting Groq's per-minute limit."""
    from openai import RateLimitError

    global _next_request_at
    settings = llm_settings()
    client, model = chat_client()
    for attempt in range(12):
        if settings["provider"] == "groq":
            with _rate_lock:
                delay = max(0.0, _next_request_at - time.monotonic())
                if delay:
                    time.sleep(delay)
                _next_request_at = time.monotonic() + config.GROQ_MIN_INTERVAL_SECONDS
        try:
            return client.chat.completions.create(model=model, **kwargs)
        except RateLimitError as exc:
            if settings["provider"] != "groq" or "per day" in str(exc).lower():
                raise
            if attempt == 11:
                raise
            header = exc.response.headers.get("retry-after") if exc.response else None
            try:
                retry_after = float(header) if header else 0.0
            except ValueError:
                retry_after = 0.0
            time.sleep(min(60.0, max(retry_after, 5.0)))
    raise RuntimeError("Groq retry limit reached")
