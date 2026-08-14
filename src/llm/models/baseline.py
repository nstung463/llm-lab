"""Public baseline-model API.

The implementation remains in ``llm.model`` for backwards compatibility while
new code should import the baseline through this module.
"""

from ..model import GPTModel, MultiHeadAttention, count_parameters

__all__ = ["GPTModel", "MultiHeadAttention", "count_parameters"]
