"""Run the golden evaluation dataset against the LangGraph app."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from evals.metrics import (
    answer_relevance,
    context_precision,
    context_recall,
    faithfulness,
    isolation_check,
    numeric_accuracy,
)
from multi_tenant_rag.config import load_settings
from multi_tenant_rag.workflow import build_graph

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "evals" / "golden.jsonl"
REPORTS_DIR = ROOT / "evals" / "reports"


@dataclass(frozen=True, slots=True)
class GoldenExample:
    question: str
    expected_answer: str
    expected_source_files: list[str]
    expected_figures: list[str]
    user: str
    intent: str
    allowed_companies: list[str]


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenExample]:
    examples: list[GoldenExample] = []
    settings = load_settings()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        allowed = list(settings.allowed_companies(payload["user"]))
        examples.append(
            GoldenExample(
                question=payload["question"],
                expected_answer=payload.get("expected_answer", ""),
                expected_source_files=list(payload.get("expected_source_files", [])),
                expected_figures=list(payload.get("expected_figures", [])),
                user=payload["user"],
                intent=payload.get("intent", "DOC"),
                allowed_companies=allowed,
            )
        )
    return examples


async def evaluate_example(graph: Any, example: GoldenExample) -> dict[str, Any]:
    question_id = hashlib.sha256(example.question.encode()).hexdigest()[:12]
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=example.question)],
            "email": example.user,
            "allowed_companies": example.allowed_companies,
            "intent": None,
            "last_user_question": None,
            "standalone_query": None,
            "retrieved": [],
            "answer": None,
            "verification": None,
            "node_timings": {},
        },
        config={"configurable": {"thread_id": f"eval-{example.user}-{question_id}"}},
    )
    answer = str(result.get("answer") or "")
    retrieved = result.get("retrieved") or []
    contexts = [str(item.get("text", "")) for item in retrieved]
    sources = [str(item.get("source", "")) for item in retrieved]
    companies = [str(item.get("company", "")) for item in retrieved]
    return {
        "question": example.question,
        "user": example.user,
        "intent": result.get("intent"),
        "answer": answer,
        "metrics": {
            "faithfulness": faithfulness(answer, contexts),
            "answer_relevance": answer_relevance(answer, example.question),
            "context_precision": context_precision(
                sources, example.expected_source_files
            ),
            "context_recall": context_recall(sources, example.expected_source_files),
            "numeric_accuracy": numeric_accuracy(answer, example.expected_figures),
            "isolation_check": isolation_check(companies, example.allowed_companies),
        },
    }


async def run_eval(limit: int | None = None) -> Path:
    load_dotenv()
    graph = build_graph()
    examples = load_golden()[:limit]
    results = [await evaluate_example(graph, example) for example in examples]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = REPORTS_DIR / f"{timestamp}.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for item in results:
        metrics = item["metrics"]
        print(
            f"{item['question'][:40]:40} "
            f"faith={metrics['faithfulness']:.2f} "
            f"iso={metrics['isolation_check']}"
        )
    print(f"Wrote {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run golden eval dataset")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run_eval(limit=args.limit))


if __name__ == "__main__":
    main()
