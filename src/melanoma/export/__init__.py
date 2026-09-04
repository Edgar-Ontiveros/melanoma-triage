"""Exportación: modelo entrenado → ONNX, más verificación numérica contra PyTorch.

Se implementa en F3. El `data_config` de la fábrica viaja junto al ONNX como
metadatos para que la API normalice exactamente igual que en entrenamiento.
"""
