"""Transformaciones de imagen (F2.1), construidas a partir del ``data_config`` de timm.

Media, desviación e interpolación salen EXCLUSIVAMENTE del diccionario que devuelve
``melanoma.models.factory.build_model``. Aquí no hay ningún valor de normalización
escrito a mano; lo vigila ``tests/test_no_hardcoded_normalization.py``.

La aumentación (albumentations) se aplica solo cuando ``train=True``. Validación recibe
únicamente redimensionado + normalización.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

# Nombre de interpolación de timm → constante de OpenCV. Es un mapeo de nombres, no un valor
# de normalización: la elección la hace el data_config del backbone.
_INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
    "area": cv2.INTER_AREA,
}

# Rango nominal de un JPEG de 8 bits, que albumentations usa para llevar píxeles a [0, 1]
# antes de restar la media. Los factores de F2.6 lo multiplican para inyectar el error.
PIXEL_MAX_8BIT = 255.0
PREPROCESS_SCALE_FACTORS = {"none": 1.0, "divide255": 255.0, "multiply255": 1.0 / 255.0}


def resolve_image_size(image_size: int | None, data_config: Mapping[str, Any]) -> int:
    """Lado de entrada: el de la config si se fijó, si no el ``input_size`` del backbone."""
    if image_size is not None:
        return int(image_size)
    return int(data_config["input_size"][-1])


def interpolation_flag(data_config: Mapping[str, Any]) -> int:
    name = str(data_config.get("interpolation", "bilinear"))
    if name not in _INTERPOLATION:
        raise ValueError(f"interpolación no soportada por OpenCV: {name!r}")
    return _INTERPOLATION[name]


def normalization(data_config: Mapping[str, Any], preprocess_scale: str = "none") -> A.Normalize:
    """``A.Normalize`` con la media/desviación del backbone.

    ``preprocess_scale`` (F2.6) altera el rango de píxel asumido: ``divide255`` hace que la
    entrada al modelo quede 255× más pequeña que lo esperado, ``multiply255`` 255× más grande.
    """
    if preprocess_scale not in PREPROCESS_SCALE_FACTORS:
        raise ValueError(
            f"preprocess_scale desconocido: {preprocess_scale!r}; "
            f"opciones: {sorted(PREPROCESS_SCALE_FACTORS)}"
        )
    factor = PREPROCESS_SCALE_FACTORS[preprocess_scale]
    return A.Normalize(
        mean=tuple(float(m) for m in data_config["mean"]),
        std=tuple(float(s) for s in data_config["std"]),
        max_pixel_value=PIXEL_MAX_8BIT * factor,
    )


def build_transforms(
    data_config: Mapping[str, Any],
    image_size: int | None,
    augment: Mapping[str, Any] | None,
    train: bool,
    preprocess_scale: str = "none",
) -> A.Compose:
    """Pipeline de albumentations para un split.

    Args:
        data_config: diccionario de ``timm.data.resolve_data_config`` (fábrica).
        image_size: lado de salida; ``None`` → ``input_size`` del backbone.
        augment: bloque ``data.augment`` de la config. Obligatorio si ``train=True``.
        train: ``True`` aplica aumentación; ``False`` solo redimensiona y normaliza.
        preprocess_scale: ``none`` salvo en el experimento F2.6.
    """
    size = resolve_image_size(image_size, data_config)
    interp = interpolation_flag(data_config)
    ops: list[A.BasicTransform] = []
    if train:
        if augment is None:
            raise ValueError("train=True requiere el bloque data.augment")
        lo, hi = (float(v) for v in augment["crop_scale"])
        ops += [
            A.RandomResizedCrop(size=(size, size), scale=(lo, hi), interpolation=interp, p=1.0),
            A.HorizontalFlip(p=float(augment["hflip_p"])),
            A.VerticalFlip(p=float(augment["vflip_p"])),
            A.Rotate(
                limit=float(augment["rotate_limit_deg"]),
                interpolation=interp,
                border_mode=cv2.BORDER_REFLECT_101,
                p=float(augment["rotate_p"]),
            ),
            A.RandomBrightnessContrast(
                brightness_limit=float(augment["brightness_limit"]),
                contrast_limit=float(augment["contrast_limit"]),
                p=float(augment["brightness_contrast_p"]),
            ),
        ]
    else:
        ops.append(A.Resize(height=size, width=size, interpolation=interp))
    ops += [normalization(data_config, preprocess_scale), ToTensorV2()]
    return A.Compose(ops)
