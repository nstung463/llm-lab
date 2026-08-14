"""Compatibility shim for the old flat training module."""

from .training.loop import evaluate, load_checkpoint, loss_for_batch, save_checkpoint, train

__all__ = ["evaluate", "load_checkpoint", "loss_for_batch", "save_checkpoint", "train"]
