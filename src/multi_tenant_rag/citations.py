"""Compatibility alias for :mod:`multi_tenant_rag.ui.citations`."""

import sys

from multi_tenant_rag.ui import citations as _implementation

sys.modules[__name__] = _implementation
