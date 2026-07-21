"""Offline ingestion pipeline for the sample earnings corpus."""

from __future__ import annotations

import argparse
import asyncio
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import chromadb
import fitz  # PyMuPDF
from chromadb.errors import NotFoundError

from multi_tenant_rag.ai.embeddings import EmbeddingClient
from multi_tenant_rag.config import Settings, load_settings
from multi_tenant_rag.rag.cache import write_index_version
from multi_tenant_rag.rag.chunking import (
    Chunk,
    ChunkMetadata,
    PageText,
    TextBlock,
    chunk_document,
    contextualize_chunk,
)
from multi_tenant_rag.rag.index import (
    COLLECTION_NAME,
)
from multi_tenant_rag.rag.index import (
    BM25Store as BM25Store,
)
from multi_tenant_rag.rag.index import (
    build_bm25_store as build_bm25_store,
)
from multi_tenant_rag.rag.index import (
    load_bm25_store as load_bm25_store,
)
from multi_tenant_rag.rag.index import (
    persist_bm25_store as persist_bm25_store,
)
from multi_tenant_rag.rag.index import (
    tokenize_for_bm25 as tokenize_for_bm25,
)


@dataclass(frozen=True, slots=True)
class IngestStats:
    companies: tuple[str, ...]
    documents: int
    chunks: int
    embed_tokens: int
    embed_cost_usd: float


def discover_pdf_paths(pdf_dir: Path) -> list[tuple[str, Path]]:
    """Return `(company, pdf_path)` pairs under `data/pdfs/<company>/`."""

    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    discovered: list[tuple[str, Path]] = []
    for company_dir in sorted(path for path in pdf_dir.iterdir() if path.is_dir()):
        company = company_dir.name.strip().lower()
        for pdf_path in sorted(company_dir.glob("*.pdf")):
            discovered.append((company, pdf_path))
    return discovered


def extract_pages(pdf_path: Path) -> list[PageText]:
    """Extract page text and layout blocks via PyMuPDF for bbox citations."""

    document = fitz.open(pdf_path)
    try:
        pages: list[PageText] = []
        for index, page in enumerate(document, start=1):
            blocks: list[TextBlock] = []
            for raw in page.get_text("blocks"):
                # blocks: x0, y0, x1, y1, text, block_no, block_type
                if len(raw) < 5:
                    continue
                x0, y0, x1, y1, text = raw[0], raw[1], raw[2], raw[3], raw[4]
                if not isinstance(text, str):
                    continue
                cleaned = text.strip()
                if not cleaned:
                    continue
                blocks.append(
                    TextBlock(
                        x0=float(x0),
                        y0=float(y0),
                        x1=float(x1),
                        y1=float(y1),
                        text=cleaned,
                    )
                )
            page_text = page.get_text("text") or ""
            pages.append(
                PageText(
                    page=index,
                    text=page_text,
                    blocks=tuple(blocks),
                )
            )
        return pages
    finally:
        document.close()


def chunk_metadata(metadata: ChunkMetadata) -> dict[str, str | int | bool]:
    payload: dict[str, str | int | bool] = {
        "company": metadata.company,
        "source": metadata.source,
        "doc_type": metadata.doc_type,
        "page": metadata.page,
        "chunk_id": metadata.chunk_id,
        "has_table": metadata.has_table,
        "pdf_relpath": metadata.pdf_relpath or f"{metadata.company}/{metadata.source}",
        "bboxes": metadata.bboxes or "[]",
    }
    if metadata.quarter is not None:
        payload["quarter"] = metadata.quarter
    if metadata.fiscal_year is not None:
        payload["fiscal_year"] = metadata.fiscal_year
    if metadata.section is not None:
        payload["section"] = metadata.section
    if metadata.speaker is not None:
        payload["speaker"] = metadata.speaker
    if metadata.quote is not None:
        payload["quote"] = metadata.quote
    return payload


async def ingest_corpus(
    settings: Settings,
    *,
    companies: Sequence[str] | None = None,
    reset: bool = False,
) -> IngestStats:
    """Chunk, embed, and persist the configured PDF corpus."""

    if settings.contextual_chunking:
        raise NotImplementedError(
            "CONTEXTUAL_CHUNKING requires the LLM client and is not enabled yet"
        )
    if companies and not reset:
        raise ValueError(
            "Partial ingestion requires --reset so Chroma and BM25 stay consistent"
        )

    allowed = {company.casefold() for company in companies} if companies else None
    pdf_paths = discover_pdf_paths(settings.pdf_dir)
    if allowed is not None:
        pdf_paths = [
            (company, path)
            for company, path in pdf_paths
            if company.casefold() in allowed
        ]
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found under {settings.pdf_dir} for ingestion")

    all_chunks: list[tuple[str, str, str, ChunkMetadata]] = []
    documents_ingested = 0
    for company, pdf_path in pdf_paths:
        pages = extract_pages(pdf_path)
        chunks = chunk_document(
            pages,
            company=company,
            source=pdf_path.name,
            min_chars=settings.chunk_min_chars,
            max_chars=settings.chunk_max_chars,
        )
        documents_ingested += 1
        for chunk in chunks:
            all_chunks.append(
                (
                    chunk.metadata.chunk_id,
                    company,
                    chunk.text,
                    chunk.metadata,
                )
            )

    embedder = EmbeddingClient(settings)
    embedding_inputs = [
        contextualize_chunk(Chunk(text=text, metadata=metadata))
        for _chunk_id, _company, text, metadata in all_chunks
    ]
    vectors = await embedder.embed_batch(embedding_inputs)

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except NotFoundError:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [chunk_id for chunk_id, _company, _text, _metadata in all_chunks]
    documents = [text for _chunk_id, _company, text, _metadata in all_chunks]
    metadatas = [
        chunk_metadata(metadata) for _chunk_id, _company, _text, metadata in all_chunks
    ]
    chroma_embeddings = cast(Any, vectors)
    chroma_metadatas = cast(Any, metadatas)
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=chroma_embeddings,
        metadatas=chroma_metadatas,
    )
    existing_ids = set(collection.get(include=[])["ids"])
    orphan_ids = sorted(existing_ids - set(ids))
    if orphan_ids:
        collection.delete(ids=orphan_ids)

    bm25_store = build_bm25_store(
        [(chunk_id, company, text) for chunk_id, company, text, _metadata in all_chunks]
    )
    persist_bm25_store(settings.bm25_index_path, bm25_store)
    write_index_version(settings.data_dir, uuid.uuid4().hex)

    stats = IngestStats(
        companies=tuple(sorted({company for company, _path in pdf_paths})),
        documents=documents_ingested,
        chunks=len(all_chunks),
        embed_tokens=embedder.usage.embed_tokens,
        embed_cost_usd=embedder.usage.embed_cost_usd,
    )
    print_stats_table(
        stats,
        Counter(str(metadata["doc_type"]) for metadata in metadatas),
    )
    return stats


def print_stats_table(
    stats: IngestStats,
    doc_type_counts: Counter[str],
) -> None:
    print("Ingestion complete")
    print(f"  companies: {', '.join(stats.companies)}")
    print(f"  documents: {stats.documents}")
    print(f"  chunks:    {stats.chunks}")
    for doc_type, count in sorted(doc_type_counts.items()):
        print(f"    {doc_type}: {count}")
    print(f"  embed tokens: {stats.embed_tokens}")
    print(f"  embed cost:   ${stats.embed_cost_usd:.6f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest the earnings PDF corpus")
    parser.add_argument(
        "--company",
        action="append",
        dest="companies",
        help="Restrict ingestion to one or more company folders",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Replace the existing Chroma collection before upserting",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    args = build_arg_parser().parse_args(argv)
    settings = load_settings()
    asyncio.run(
        ingest_corpus(
            settings,
            companies=args.companies,
            reset=args.reset,
        )
    )


if __name__ == "__main__":
    main()
