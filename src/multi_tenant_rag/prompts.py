"""Compatibility alias for :mod:`multi_tenant_rag.ai.prompts`."""

import sys

from multi_tenant_rag.ai import prompts as _implementation

sys.modules[__name__] = _implementation
