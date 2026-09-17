"""Scheduler de tasa de aprendizaje: cosine annealing por paso con calentamiento lineal."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """Multiplicador del lr en ``step``: sube linealmente durante ``warmup_steps`` y luego
    decae en coseno hasta 0 en ``total_steps``."""
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def cosine_with_warmup(optimizer: Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    return LambdaLR(
        optimizer, lambda step: cosine_with_warmup_factor(step, warmup_steps, total_steps)
    )
