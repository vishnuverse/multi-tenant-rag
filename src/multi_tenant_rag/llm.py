"""Compatibility alias for :mod:`multi_tenant_rag.ai.client`."""

import sys

from multi_tenant_rag.ai import client as _implementation

sys.modules[__name__] = _implementation
