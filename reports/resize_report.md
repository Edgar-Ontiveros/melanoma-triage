# Redimensionado ISIC 2020

Generado por `scripts/resize_images.py`. Lado largo 512 px, relación de aspecto preservada, JPEG calidad 95, remuestreo Lanczos. Salida: `data/processed/isic2020_512`.

- Imágenes: 33,126
- Tamaño total en disco: **1.56 GB** (originales: 25.76 GB, factor 16.5x)
- Imágenes que ya eran ≤ 512 px y no se redujeron: 0
- Dimensiones resultantes: lado largo mín 512, máx 512
- Tiempo: 7.8 min con 12 procesos

Columnas agregadas al manifiesto: `sha256_resized`, `relpath_resized`, `width_resized`, `height_resized`, `bytes_resized`.
