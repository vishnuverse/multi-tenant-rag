# multi-tenant-rag

Multi-tenant RAG over earnings PDFs. Each user can only query companies they are allowed to see.

**Stack:** Chainlit · LangGraph · OpenRouter · Chroma

## Run

```sh
cp .env.example .env
# set OPENROUTER_API_KEY and CHAINLIT_AUTH_SECRET

uv sync
uv run python -m multi_tenant_rag.ingest --reset
uv run chainlit run src/multi_tenant_rag/app.py --host 127.0.0.1 --port 8005
```

## Demo logins

Any password works (for example `demo`):

| Email | Access |
|-------|--------|
| `alice@example.com` | apple |
| `bob@example.com` | microsoft, google |
| `charlie@example.com` | meta, amazon |

## Example questions

- Alice: `What were Apple's total net sales in FY25 Q1?`
- Bob: `What did Microsoft say about cloud revenue?`
- Charlie: `What was Amazon's net sales in Q4 2024?`
