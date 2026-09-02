from __future__ import annotations


class WorkflowIntegrationNotAvailable(RuntimeError):
    """Raised when a later milestone tries to use workflow integration before it exists."""
