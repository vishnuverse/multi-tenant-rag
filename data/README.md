# Data

- `pdfs/` — sample earnings corpus and provenance (`pdfs/README.md`)
- `chroma/`, `bm25_index.pkl`, `index_version.txt`, `logs/` — generated locally; gitignored

Run ingest before chatting:

```sh
uv run python -m multi_tenant_rag.ingest --reset
```
