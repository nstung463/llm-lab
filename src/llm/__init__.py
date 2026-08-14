"""Small, inspectable GPT components used by the learning lab."""

from .config import ModelConfig, TrainingConfig
from .models.baseline import GPTModel, MultiHeadAttention, count_parameters

__all__ = ["GPTModel", "ModelConfig", "MultiHeadAttention", "TrainingConfig", "count_parameters"]
