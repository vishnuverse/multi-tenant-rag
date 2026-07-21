"""Compatibility alias for :mod:`multi_tenant_rag.workflow.graph`."""

from __future__ import annotations

import sys

from multi_tenant_rag.workflow import graph as _implementation

sys.modules[__name__] = _implementation
