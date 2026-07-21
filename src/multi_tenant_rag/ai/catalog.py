"""Live, validated OpenRouter chat-model catalog."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal, DecimalException, Overflow, Underflow, localcontext
from math import isfinite
from typing import Any

import httpx

from multi_tenant_rag.config import Settings

_CATALOG_PARAMS = {
    "input_modalities": "text",
    "output_modalities": "text",
    "supported_parameters": "response_format",
    "sort": "pricing-low-to-high",
}


@dataclass(frozen=True, slots=True)
class ChatModel:
    """Canonical compatible chat model and its USD cost per million tokens."""

    id: str
    name: str
    prompt_price_per_million: float
    completion_price_per_million: float

    @property
    def label(self) -> str:
        """Return the human-readable dropdown label."""

        return (
            f"{self.name} — ${self.prompt_price_per_million:g} input / "
            f"${self.completion_price_per_million:g} output per 1M"
        )


@dataclass(frozen=True, slots=True)
class CatalogResult:
    """A catalog snapshot, including whether its offline fallback was used."""

    models: tuple[ChatModel, ...]
    used_fallback: bool
    stale: bool = False
    outage: bool = False


# Prices are OpenRouter's approved model-level prices, not a provider override.
# Default/fallback order: preferred demo default first, then cheaper/faster options.
FALLBACK_MODELS = (
    ChatModel(
        id="openai/gpt-4.1-mini",
        name="OpenAI: GPT-4.1 Mini",
        prompt_price_per_million=0.4,
        completion_price_per_million=1.6,
    ),
    ChatModel(
        id="google/gemini-2.5-flash-lite",
        name="Google: Gemini 2.5 Flash Lite",
        prompt_price_per_million=0.1,
        completion_price_per_million=0.4,
    ),
    ChatModel(
        id="moonshotai/kimi-k2.5",
        name="MoonshotAI: Kimi K2.5",
        prompt_price_per_million=0.57,
        completion_price_per_million=2.85,
    ),
)


def _string_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _price_per_million(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return None
    try:
        per_token = Decimal(str(value))
        if not per_token.is_finite() or per_token < 0:
            return None
        with localcontext() as context:
            context.traps[Overflow] = True
            context.traps[Underflow] = True
            per_million = per_token * Decimal(1_000_000)
        price = float(per_million)
    except (DecimalException, OverflowError, ValueError):
        return None
    if (
        not per_million.is_finite()
        or not isfinite(price)
        or price < 0
        or (per_token != 0 and (per_million.is_zero() or price == 0))
    ):
        return None
    return price


def _parse_model(raw: object) -> ChatModel | None:
    if not isinstance(raw, Mapping):
        return None
    model_id = raw.get("id")
    name = raw.get("name")
    architecture = raw.get("architecture")
    pricing = raw.get("pricing")
    parameters = _string_list(raw.get("supported_parameters"))
    if (
        not isinstance(model_id, str)
        or not model_id.strip()
        or not isinstance(architecture, Mapping)
        or not isinstance(pricing, Mapping)
        or "response_format" not in parameters
    ):
        return None
    inputs = _string_list(architecture.get("input_modalities"))
    outputs = _string_list(architecture.get("output_modalities"))
    if "text" not in inputs or "text" not in outputs:
        return None
    prompt = _price_per_million(pricing.get("prompt"))
    completion = _price_per_million(pricing.get("completion"))
    if prompt is None or completion is None:
        return None
    display_name = name.strip() if isinstance(name, str) and name.strip() else model_id
    return ChatModel(
        id=model_id.strip(),
        name=display_name,
        prompt_price_per_million=prompt,
        completion_price_per_million=completion,
    )


def parse_chat_models(payload: object) -> tuple[ChatModel, ...]:
    """Parse, defensively filter, and deterministically sort a models response."""

    if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
        raise ValueError("OpenRouter models response must contain a data list")
    models = [
        model for raw in payload["data"] if (model := _parse_model(raw)) is not None
    ]
    models.sort(
        key=lambda model: (
            model.prompt_price_per_million + model.completion_price_per_million,
            model.id,
        )
    )
    return tuple(models)


class ModelCatalog:
    """Concurrency-safe TTL cache around OpenRouter's models endpoint."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._client = client
        self._clock = clock
        self._lock = asyncio.Lock()
        self._cached: CatalogResult | None = None
        self._last_successful: tuple[ChatModel, ...] | None = None
        self._expires_at = 0.0

    def invalidate(self) -> None:
        """Clear the cached snapshot, primarily for deterministic tests."""

        self._cached = None
        self._last_successful = None
        self._expires_at = 0.0

    def lookup(self, model_id: str) -> ChatModel | None:
        """Read a model from the current snapshot without network activity."""

        canonical_id = model_id.strip()
        if not canonical_id:
            return None
        models = self._cached.models if self._cached is not None else FALLBACK_MODELS
        return next((model for model in models if model.id == canonical_id), None)

    async def get_models(self) -> CatalogResult:
        """Return a fresh/live catalog or the compatible built-in fallback."""

        now = self._clock()
        if self._cached is not None and now < self._expires_at:
            return self._cached
        async with self._lock:
            now = self._clock()
            if self._cached is not None and now < self._expires_at:
                return self._cached
            result = await self._load()
            loaded_at = self._clock()
            self._cached = result
            self._expires_at = loaded_at + self._settings.model_catalog_ttl_seconds
            return result

    async def _load(self) -> CatalogResult:
        try:
            models = await self._fetch_live()
            if not models:
                raise ValueError("OpenRouter returned no compatible chat models")
            self._last_successful = models
            return CatalogResult(models=models, used_fallback=False)
        except (httpx.HTTPError, ArithmeticError, TypeError, ValueError):
            if self._last_successful is not None:
                return CatalogResult(
                    models=self._last_successful,
                    used_fallback=False,
                    stale=True,
                    outage=True,
                )
            return CatalogResult(
                models=FALLBACK_MODELS,
                used_fallback=True,
                outage=True,
            )

    async def _fetch_live(self) -> tuple[ChatModel, ...]:
        headers: dict[str, str] = {}
        if self._settings.openrouter_api_key:
            headers["Authorization"] = f"Bearer {self._settings.openrouter_api_key}"
        url = f"{self._settings.openrouter_base_url.rstrip('/')}/models"
        if self._client is not None:
            response = await self._client.get(
                url,
                params=_CATALOG_PARAMS,
                headers=headers,
                timeout=10.0,
            )
        else:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    params=_CATALOG_PARAMS,
                    headers=headers,
                    timeout=10.0,
                )
        response.raise_for_status()
        payload: Any = response.json()
        return parse_chat_models(payload)


_catalogs: dict[tuple[str, str, int], ModelCatalog] = {}


def catalog_for(settings: Settings) -> ModelCatalog:
    """Return the process-local catalog cache for a configuration."""

    key = (
        settings.openrouter_base_url,
        settings.openrouter_api_key,
        settings.model_catalog_ttl_seconds,
    )
    catalog = _catalogs.get(key)
    if catalog is None:
        catalog = ModelCatalog(settings)
        _catalogs[key] = catalog
    return catalog


def invalidate_model_catalog() -> None:
    """Invalidate every process-local catalog instance."""

    for catalog in _catalogs.values():
        catalog.invalidate()
    _catalogs.clear()
