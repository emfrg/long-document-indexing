from __future__ import annotations


class FoundryEvaluationNotAvailable(RuntimeError):
    """Raised when Foundry evaluation is requested before the adapter is implemented."""
