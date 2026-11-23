"""Memory-efficient backprop through large output heads."""

from membackprop.backprop import MemoryEfficientLinear, chunked_head_apply

__all__ = ["MemoryEfficientLinear", "chunked_head_apply"]
