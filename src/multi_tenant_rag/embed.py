"""Compatibility alias for :mod:`multi_tenant_rag.ai.embeddings`."""

import sys

from multi_tenant_rag.ai import embeddings as _implementation

sys.modules[__name__] = _implementation
