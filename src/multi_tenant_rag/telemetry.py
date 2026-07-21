"""Runtime telemetry for per-turn logging and usage counters."""

from __future__ import annotations

import json
import time
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any

from multi_tenant_rag.config import Settings, load_settings


@dataclass(slots=True)
class UsageCounters:
    """Aggregate usage for a process or ingestion run."""

    embed_tokens: int = 0
    embed_cost_usd: float = 0.0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_cost_usd: float = 0.0
    cost_complete: bool = True
    unknown_cost_model_ids: set[str] = field(default_factory=set)
    missing_usage_model_ids: set[str] = field(default_factory=set)

    def record_embedding(self, *, tokens: int, cost_usd: float) -> None:
        self.embed_tokens += tokens
        self.embed_cost_usd += cost_usd

    def record_llm(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float | None,
        model_id: str,
    ) -> None:
        self.llm_input_tokens += input_tokens
        self.llm_output_tokens += output_tokens
        if cost_usd is None:
            self.cost_complete = False
            self.unknown_cost_model_ids.add(model_id)
        else:
            self.llm_cost_usd += cost_usd

    def record_missing_llm_usage(self, *, model_id: str) -> None:
        """Mark a model call whose provider omitted token usage."""

        self.cost_complete = False
        self.missing_usage_model_ids.add(model_id)

    @property
    def total_cost_usd(self) -> float:
        return self.embed_cost_usd + self.llm_cost_usd


@dataclass(slots=True)
class TurnRecord:
    """Structured telemetry for one assistant turn."""

    trace_id: str
    thread_id: str
    user: str
    selected_model: str
    intent: str | None
    standalone_query: str | None
    chunk_ids: list[str]
    rerank_scores: list[float]
    verification: dict[str, Any] | None
    tokens_by_node: dict[str, int]
    cost_usd_by_node: dict[str, float]
    latency_ms_by_node: dict[str, float]
    cache_hit: bool = False
    total_cost_usd: float = 0.0
    cost_complete: bool = True
    unknown_cost_model_ids: list[str] = field(default_factory=list)
    missing_usage_model_ids: list[str] = field(default_factory=list)
    error: str | None = None
    total_latency_ms: float = 0.0


@dataclass(slots=True)
class TurnTelemetry:
    """Capture per-node timings and write JSONL turn records."""

    settings: Settings
    counters: UsageCounters = field(default_factory=UsageCounters)
    _node_latencies: dict[str, float] = field(default_factory=dict, repr=False)
    _turn_started_at: float = field(default=0.0, repr=False)

    def start_turn(self) -> None:
        self._node_latencies = {}
        self._turn_started_at = time.perf_counter()

    def record_node(self, node: str, *, latency_ms: float) -> None:
        self._node_latencies[node] = latency_ms

    def write_turn(
        self,
        *,
        thread_id: str,
        user: str,
        selected_model: str,
        intent: str | None,
        standalone_query: str | None,
        chunk_ids: list[str],
        rerank_scores: list[float],
        verification: dict[str, Any] | None,
        cache_hit: bool,
        node_timings: dict[str, float] | None = None,
        error: str | None = None,
    ) -> TurnRecord:
        total_latency_ms = (time.perf_counter() - self._turn_started_at) * 1000
        record = TurnRecord(
            trace_id=str(uuid.uuid4()),
            thread_id=thread_id,
            user=user,
            selected_model=selected_model,
            intent=intent,
            standalone_query=standalone_query,
            chunk_ids=chunk_ids,
            rerank_scores=rerank_scores,
            verification=verification,
            tokens_by_node={
                "embed": self.counters.embed_tokens,
                "llm": self.counters.llm_input_tokens + self.counters.llm_output_tokens,
            },
            cost_usd_by_node={
                "embed": self.counters.embed_cost_usd,
                "llm": self.counters.llm_cost_usd,
            },
            latency_ms_by_node=dict(node_timings or self._node_latencies),
            cache_hit=cache_hit,
            total_cost_usd=self.counters.total_cost_usd,
            cost_complete=self.counters.cost_complete,
            unknown_cost_model_ids=sorted(self.counters.unknown_cost_model_ids),
            missing_usage_model_ids=sorted(self.counters.missing_usage_model_ids),
            error=error,
            total_latency_ms=total_latency_ms,
        )
        self._append_jsonl(record)
        if self.settings.langfuse_enabled:
            pass
        return record

    def _append_jsonl(self, record: TurnRecord) -> None:
        self.settings.logs_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.logs_dir / "turns.jsonl"
        payload = asdict(record)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


_CURRENT_COUNTERS: ContextVar[UsageCounters | None] = ContextVar(
    "usage_counters",
    default=None,
)


def usage_counters() -> UsageCounters:
    counters = _CURRENT_COUNTERS.get()
    if counters is None:
        counters = UsageCounters()
        _CURRENT_COUNTERS.set(counters)
    return counters


def reset_usage_counters() -> UsageCounters:
    counters = UsageCounters()
    _CURRENT_COUNTERS.set(counters)
    return counters


def turn_telemetry(settings: Settings | None = None) -> TurnTelemetry:
    return TurnTelemetry(
        settings or load_settings(),
        counters=usage_counters(),
    )
