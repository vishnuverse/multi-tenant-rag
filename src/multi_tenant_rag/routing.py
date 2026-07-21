"""Compatibility alias for :mod:`multi_tenant_rag.workflow.routing`."""

import sys

from multi_tenant_rag.workflow import routing as _implementation

sys.modules[__name__] = _implementation
