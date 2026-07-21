"""Compatibility alias for :mod:`multi_tenant_rag.rag.verification`."""

import sys

from multi_tenant_rag.rag import verification as _implementation

sys.modules[__name__] = _implementation
