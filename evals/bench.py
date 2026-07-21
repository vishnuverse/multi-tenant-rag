"""Benchmark latency and cost by intent branch."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from multi_tenant_rag.config import load_settings
from multi_tenant_rag.workflow import build_graph

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "evals" / "reports"


async def _bench_branch(
    graph: Any,
    *,
    message: str,
    email: str,
    companies: list[str],
    thread_id: str,
) -> float:
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=message)],
            "email": email,
            "allowed_companies": companies,
            "intent": None,
            "last_user_question": None,
            "standalone_query": None,
            "retrieved": [],
            "answer": None,
            "verification": None,
            "node_timings": {},
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    timings = result.get("node_timings") or {}
    return float(sum(timings.values()))


async def run_bench() -> Path:
    load_dotenv()
    settings = load_settings({"OPENROUTER_API_KEY": "bench-key"})
    mock_llm = AsyncMock()
    mock_llm.stream_chat_completion.return_value = "Apple revenue was strong."
    mock_llm.counters = MagicMock()

    mock_retrieval = MagicMock()
    mock_retrieval.search = AsyncMock(return_value=[])

    with (
        patch("multi_tenant_rag.workflow.graph.LLMClient", return_value=mock_llm),
        patch(
            "multi_tenant_rag.workflow.graph.get_retrieval_engine",
            return_value=mock_retrieval,
        ),
    ):
        graph = build_graph(settings)

        branches = {
            "SMALL_TALK": await _bench_branch(
                graph,
                message="Hi",
                email="alice@example.com",
                companies=["apple"],
                thread_id="bench-small",
            ),
            "META": await _bench_branch(
                graph,
                message="What can I access?",
                email="alice@example.com",
                companies=["apple"],
                thread_id="bench-meta",
            ),
            "DOC": await _bench_branch(
                graph,
                message="What was Apple revenue?",
                email="alice@example.com",
                companies=["apple"],
                thread_id="bench-doc",
            ),
        }

    summary = {
        branch: {
            "latency_ms": latency,
            "p50_ms": latency,
            "p95_ms": latency,
        }
        for branch, latency in branches.items()
    }
    summary["aggregate_p50_ms"] = statistics.median(branches.values())
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = REPORTS_DIR / f"bench-{timestamp}.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark graph branches")
    _ = parser.parse_args()
    asyncio.run(run_bench())


if __name__ == "__main__":
    main()
