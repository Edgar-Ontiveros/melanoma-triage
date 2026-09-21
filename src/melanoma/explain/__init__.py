"""Explicabilidad (F5): CAM en numpy puro para la API de F6 y, en submódulos con torch, la
extracción de mapas de activación, el Grad-CAM de referencia, la prueba de cordura, las
máscaras automáticas de lesión y los artefactos sintéticos.

Importar ``melanoma.explain`` no carga torch: solo ``cam`` (numpy). Los submódulos
``features``, ``sanity``, ``masks`` y ``perturb`` requieren el grupo ``train``.
"""

from melanoma.explain.cam import cam_from_features, energy_fraction, explain, upsample

__all__ = ["cam_from_features", "energy_fraction", "explain", "upsample"]
