"""Semillas y determinismo.

Qué se garantiza y qué no, con honestidad:

- SÍ: con la misma semilla y la misma config, el orden de los datos, la inicialización
  de pesos y los workers del DataLoader son idénticos entre corridas.
- NO: reproducibilidad bit a bit universal. En GPU, algunas operaciones de cuDNN
  (p. ej. ciertas convoluciones hacia atrás y `scatter_add`) siguen siendo no
  deterministas aunque se pida lo contrario, y el modo determinista cuesta velocidad.
  Con `warn_only=True` PyTorch avisa en lugar de fallar cuando no existe kernel
  determinista para una operación.
"""

from __future__ import annotations

import lightning as L
import torch


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Fija Python, NumPy, PyTorch y los workers del DataLoader.

    Args:
        seed: semilla entera.
        deterministic: si ``True`` activa ``torch.use_deterministic_algorithms`` en modo
            ``warn_only`` y desactiva el benchmark de cuDNN.

    Returns:
        La semilla usada, para registrarla en la config de la corrida.
    """
    L.seed_everything(seed, workers=True, verbose=False)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return seed
