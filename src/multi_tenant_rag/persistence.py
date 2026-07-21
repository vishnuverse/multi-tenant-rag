"""Compatibility alias for :mod:`multi_tenant_rag.storage.persistence`."""

import sys

from multi_tenant_rag.storage import persistence as _implementation

sys.modules[__name__] = _implementation
