"""Servicio de inferencia (FastAPI + onnxruntime).

Regla estructural: este paquete NO importa torch, timm ni lightning, ni directa ni
transitivamente. Lo vigila `tests/test_api_no_torch_import.py` y el límite de 400 MB
de la imagen Docker en CI. Los endpoints se implementan en F3.
"""
