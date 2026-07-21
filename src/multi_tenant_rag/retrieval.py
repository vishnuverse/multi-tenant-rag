"""Compatibility alias for :mod:`multi_tenant_rag.rag.retrieval`."""

import sys

from multi_tenant_rag.rag import retrieval as _implementation

sys.modules[__name__] = _implementation
