"""Servicio de inferencia (FastAPI + onnxruntime).

Regla estructural: este paquete NO importa torch, timm ni lightning, ni directa ni
transitivamente. Tampoco importa el paquete `melanoma` (arrastraría hydra y omegaconf):
sus dependencias son exactamente el grupo `api` de pyproject.toml. Lo vigilan
`tests/test_api_dependency_isolation.py` y el límite de 400 MB de la imagen Docker en CI.
Los endpoints se implementan en F3.
"""
