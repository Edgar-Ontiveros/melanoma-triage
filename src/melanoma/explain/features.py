"""Parte con torch de F5.1: mapas de activación, pesos de la capa lineal y Grad-CAM de
referencia (``pytorch-grad-cam``) para verificar la equivalencia con ``cam.py``.

Capa exacta: ``backbone.bn2`` de timm (``BatchNormAct2d`` = BatchNorm + SiLU tras
``conv_head``), que es lo último antes del pooling global. Para EfficientNetV2-S a 224 px
su salida mide ``1280 × 7 × 7`` (stride total 32). ``forward_features`` de timm devuelve
exactamente ese tensor, y la cabeza del proyecto hace ``fc(dropout(mean(A, (2, 3))))``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn

from melanoma.data.transforms import eval_geometry, interpolation_flag, normalization
from melanoma.models.factory import BackboneClassifier

TARGET_LAYER = "backbone.bn2"


def classifier_of(model: nn.Module) -> BackboneClassifier:
    """Desenvuelve un ``LitBinaryClassifier`` (atributo ``model``) si hace falta."""
    inner = getattr(model, "model", model)
    if not isinstance(inner, BackboneClassifier):
        raise TypeError(f"se esperaba BackboneClassifier; llegó {type(inner).__name__}")
    return inner


def target_layer(model: nn.Module) -> nn.Module:
    """La capa cuya salida entra al pooling global (``backbone.bn2`` en timm EfficientNet)."""
    return classifier_of(model).backbone.bn2


def linear_weight(model: nn.Module) -> np.ndarray:
    """Pesos ``w_k`` de la capa lineal de una salida, como vector ``[C]`` en numpy."""
    fc = classifier_of(model).head.fc
    if fc.out_features != 1:
        raise ValueError("el contrato CAM asume una sola salida (logit binario)")
    return fc.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)


@torch.no_grad()
def extract_features(model: nn.Module, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """``(features[B, C, H, W], logits[B])`` para un lote normalizado ``x``.

    Los logits se calculan con la misma cabeza del modelo a partir de los mapas, de modo que
    el llamador pueda comprobar que coinciden con ``model(x)``.
    """
    clf = classifier_of(model)
    clf.eval()
    feats = clf.backbone.forward_features(x)
    logits = clf.head(feats.mean(dim=(2, 3)))
    return feats.float().cpu().numpy(), logits.squeeze(1).float().cpu().numpy()


def gradcam_reference(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Grad-CAM de ``pytorch-grad-cam`` sobre ``TARGET_LAYER`` para el logit único.

    Devuelve ``[B, H_in, W_in]`` en ``[0, 1]`` (la librería interpola al tamaño de entrada y
    normaliza al máximo por imagen). Se importa aquí, no arriba, para que el resto del paquete
    no dependa de la librería de referencia.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    clf = classifier_of(model)
    clf.eval()
    with GradCAM(model=clf, target_layers=[target_layer(clf)]) as cam:
        maps = cam(input_tensor=x, targets=[ClassifierOutputTarget(0)] * x.shape[0])
    return np.asarray(maps, dtype=np.float32)


def crop_transform(
    data_config: Mapping[str, Any], image_size: int, val_resize: str = "center_crop"
) -> A.Compose:
    """Solo la geometría de validación: devuelve la imagen ``uint8`` que ve el modelo, sin
    normalizar. Sobre ese recorte se calculan máscaras y se inyectan artefactos (F5.3, F5.4),
    para que queden alineados con el CAM interpolado a ``image_size``."""
    return A.Compose(eval_geometry(image_size, interpolation_flag(data_config), val_resize))


def tensor_transform(data_config: Mapping[str, Any]) -> A.Compose:
    """Normalización + ``ToTensorV2`` sobre un recorte ya del tamaño del modelo."""
    return A.Compose([normalization(data_config), ToTensorV2()])


def to_batch(crops: list[np.ndarray], tensor_tf: A.Compose) -> torch.Tensor:
    return torch.stack([tensor_tf(image=c)["image"] for c in crops])
