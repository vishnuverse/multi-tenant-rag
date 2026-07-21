"""Stable ingestion module and ``python -m`` entry point."""

from __future__ import annotations

import sys

from multi_tenant_rag.rag import ingest as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
