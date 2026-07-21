"""Compatibility alias for :mod:`multi_tenant_rag.rag.chunking`."""

import sys

from multi_tenant_rag.rag import chunking as _implementation

sys.modules[__name__] = _implementation
