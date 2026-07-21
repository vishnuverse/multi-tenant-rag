"""Compatibility alias for :mod:`multi_tenant_rag.rag.cache`."""

import sys

from multi_tenant_rag.rag import cache as _implementation

sys.modules[__name__] = _implementation
