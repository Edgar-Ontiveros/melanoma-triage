"""Fábrica de backbones.

Contrato: `build_model(backbone, pretrained, num_classes, dropout)` devuelve
`(modelo, data_config)`. El `data_config` sale de `timm.data.resolve_data_config`
y es la ÚNICA fuente de normalización del proyecto (`mean`, `std`, `input_size`,
`interpolation`). Ninguna otra parte del código debe definir esos valores.

La cabeza es idéntica para todos los backbones y se define una sola vez aquí:
pooling global promedio (lo hace timm con `num_classes=0`) → dropout → lineal.
"""

from __future__ import annotations

from typing import Any

import timm
import torch
from torch import nn


class ClassifierHead(nn.Module):
    """Cabeza común: dropout → capa lineal. Recibe features ya pooleadas."""

    def __init__(self, in_features: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(x))


class BackboneClassifier(nn.Module):
    """Backbone timm (sin clasificador propio, con pooling global) + `ClassifierHead`."""

    def __init__(self, backbone: nn.Module, head: ClassifierHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_model(
    backbone: str,
    pretrained: bool,
    num_classes: int,
    dropout: float,
) -> tuple[nn.Module, dict[str, Any]]:
    """Construye el modelo y resuelve la configuración de datos del backbone.

    Args:
        backbone: nombre del modelo en timm (p. ej. ``tf_efficientnetv2_s.in21k_ft_in1k``).
        pretrained: si ``True`` descarga/usa los pesos preentrenados de timm.
        num_classes: salidas de la capa lineal (1 para clasificación binaria con logits).
        dropout: probabilidad de dropout antes de la capa lineal.

    Returns:
        ``(model, data_config)``. ``data_config`` contiene al menos ``input_size``,
        ``mean``, ``std`` e ``interpolation``, resueltos por timm a partir del
        ``pretrained_cfg`` del backbone.
    """
    trunk = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
    data_config = timm.data.resolve_data_config({}, model=trunk)
    head = ClassifierHead(trunk.num_features, num_classes=num_classes, dropout=dropout)
    return BackboneClassifier(trunk, head), dict(data_config)
