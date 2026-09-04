"""Paquete `melanoma`: pipeline de entrenamiento, evaluación y exportación.

Subpaquetes:
- `data`: manifiestos, splits y DataModule (F1).
- `models`: fábrica de backbones timm con cabeza común (F0).
- `train`: LightningModule y bucle de entrenamiento (F0 mínimo, F2 completo).
- `eval`: métricas, curvas y calibración (F2).
- `explain`: Grad-CAM y mapas de atención (F4).
- `export`: exportación a ONNX y verificación numérica (F3).
- `utils`: semillas y utilidades transversales.
"""

__version__ = "0.0.1"
