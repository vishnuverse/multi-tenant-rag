"""Compatibility alias for :mod:`multi_tenant_rag.ui.pdf`."""

import sys

from multi_tenant_rag.ui import pdf as _implementation

sys.modules[__name__] = _implementation
