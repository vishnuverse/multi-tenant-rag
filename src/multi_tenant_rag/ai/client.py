"""Async OpenRouter chat client shared by graph nodes."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from openai import APIError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from multi_tenant_rag.ai import policy
from multi_tenant_rag.ai.catalog import catalog_for
from multi_tenant_rag.config import Settings
from multi_tenant_rag.streaming import emit_token
from multi_tenant_rag.telemetry import UsageCounters, usage_counters

_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    re.IGNORECASE | re.DOTALL,
)


class ChatModelAPIError(RuntimeError):
    """A sanitized OpenRouter chat failure tied to one canonical model ID."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(f"Chat completion failed for model {model_id}")


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object, stripping optional markdown fences."""

    text = raw.strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON response must be an object")
    return parsed


@dataclass
class LLMClient:
    """Thin wrapper around the OpenRouter chat completions API."""

    settings: Settings
    counters: UsageCounters | None = None
    _client: AsyncOpenAI | None = field(default=None, repr=False, compare=False)

    async def chat_completion(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        model: str | None = None,
        response_format: Mapping[str, str] | None = None,
    ) -> str:
        client = self._client or self._build_client()
        self._client = client
        resolved_model = policy.resolve_chat_model(model, self.settings)
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": list(messages),
        }
        if response_format is not None:
            kwargs["response_format"] = dict(response_format)

        try:
            response = await client.chat.completions.create(**kwargs)
        except APIError as exc:
            raise ChatModelAPIError(resolved_model) from exc
        choice = response.choices[0].message
        content = choice.content or ""
        self._record_usage(resolved_model, response.usage)
        return content

    async def stream_chat_completion(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        model: str | None = None,
    ) -> str:
        """Stream tokens to the active UI sink and return the full text."""

        client = self._client or self._build_client()
        self._client = client
        resolved_model = policy.resolve_chat_model(model, self.settings)
        chunks: list[str] = []
        usage: Any | None = None
        try:
            stream = await client.chat.completions.create(
                model=resolved_model,
                messages=list(messages),
                stream=True,
                stream_options={"include_usage": True},
            )
            async for event in stream:
                if event.usage is not None:
                    usage = event.usage
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                piece = delta.content or ""
                if piece:
                    chunks.append(piece)
                    await emit_token(piece)
        except APIError as exc:
            raise ChatModelAPIError(resolved_model) from exc
        content = "".join(chunks)
        self._record_usage(resolved_model, usage)
        return content

    async def chat_json(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for _attempt in range(2):
            raw = await self.chat_completion(
                messages,
                model=model,
                response_format={"type": "json_object"},
            )
            try:
                return parse_json_object(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
        raise ValueError(
            f"Failed to parse LLM JSON response: {last_error}"
        ) from last_error

    def _record_usage(self, model: str, usage: Any | None) -> None:
        usage_counters_for_call = self.counters or usage_counters()
        if usage is None:
            usage_counters_for_call.record_missing_llm_usage(model_id=model)
            return
        input_tokens = getattr(usage, "prompt_tokens", None) or 0
        output_tokens = getattr(usage, "completion_tokens", None) or 0
        catalog_model = catalog_for(self.settings).lookup(model)
        pricing = (
            (
                catalog_model.prompt_price_per_million,
                catalog_model.completion_price_per_million,
            )
            if catalog_model is not None
            else None
        )
        cost_usd = (
            (int(input_tokens) / 1_000_000) * pricing[0]
            + (int(output_tokens) / 1_000_000) * pricing[1]
            if pricing is not None
            else None
        )
        usage_counters_for_call.record_llm(
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            cost_usd=cost_usd,
            model_id=model,
        )

    def _build_client(self) -> AsyncOpenAI:
        headers: dict[str, str] = {}
        if self.settings.openrouter_app_url:
            headers["HTTP-Referer"] = self.settings.openrouter_app_url
        if self.settings.openrouter_app_title:
            headers["X-Title"] = self.settings.openrouter_app_title
        return AsyncOpenAI(
            api_key=self.settings.require_openrouter_api_key(),
            base_url=self.settings.openrouter_base_url,
            default_headers=headers or None,
        )


def build_openrouter_client(settings: Settings) -> AsyncOpenAI:
    """Build an AsyncOpenAI client pointed at OpenRouter."""

    return LLMClient(settings)._build_client()
