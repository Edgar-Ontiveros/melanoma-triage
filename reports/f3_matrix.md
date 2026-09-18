# F3 — Matriz de modelos: comparación controlada sobre validación

Generado por `scripts/f3_report.py`. Lo fijo (idéntico a B1): cabeza de la fábrica, BCE con `pos_weight` automático, AdamW 1e-4, cosine con warmup, 15 épocas con early stopping por AUPRC (paciencia 5), aumentación de F2, lado corto + CenterCrop en validación, **lote efectivo 128** (acumulación de gradiente), semillas 0/1/2, 16-mixed. Lo variable: backbone y resolución. Lo vigila `tests/test_f3_matrix.py`. `test.txt` no se toca.

## 1. Tabla principal

| config | backbone | px | semillas | AUC-ROC | AUPRC | sens@spec 0.90 | sens@spec 0.95 | spec@sens 0.90 | spec@sens 0.95 |
|:--|:--|--:|--:|:--|:--|:--|:--|:--|:--|
| **B1** | `resnet50.a1_in1k` | 224 | 3 | 0.778 (0.774–0.782) | 0.077 (0.076–0.079) | 0.402 (0.364–0.455) | 0.261 (0.239–0.284) | 0.424 (0.395–0.460) | 0.313 (0.262–0.363) |
| **A1** | `tf_efficientnetv2_s.in21k_ft_in1k` | 224 | 0 | n/d | n/d | n/d | n/d | n/d | n/d |
| **M1** | `tf_efficientnetv2_s.in21k_ft_in1k` | 384 | 0 | n/d | n/d | n/d | n/d | n/d | n/d |
| **M2** | `convnext_tiny.fb_in22k_ft_in1k_384` | 384 | 0 | n/d | n/d | n/d | n/d | n/d | n/d |

Media y rango (mín–máx) sobre semillas; cada corrida evaluada sobre `val.txt` con IC por paciente en su `metrics.json`.

## 2. Comparaciones aisladas

- **Efecto de la resolución, mismo backbone (224 → 384)** (M1 − A1): _pendiente_
- **Efecto de la arquitectura a 224 px (ResNet50 → EfficientNetV2-S)** (A1 − B1): _pendiente_
- **Efecto de la arquitectura a 384 px (EfficientNetV2-S → ConvNeXt-T)** (M2 − M1): _pendiente_

Umbral: diferencias de medias menores que 0.02 no distinguen nada (dos ejecuciones de la misma semilla difirieron 0.016 de AUC-ROC en F2).

## 3. Ablación de resolución por semilla

_pendiente_

## 4. Tiempos y utilización de GPU

| corrida | commit | épocas (mejor) | min entrenamiento | min/época | espera de datos | GPU media | colapso |
|:--|:--|:--|--:|--:|--:|--:|:--|
| `resnet50-224-s0-f431f11` | `f431f11` | 11 (5) | 22 | n/d | n/d | n/d | no |
| `resnet50-224-s1-bcdbbf9` | `bcdbbf9` | 12 (6) | 24 | n/d | n/d | n/d | no |
| `resnet50-224-s2-bcdbbf9` | `bcdbbf9` | 13 (7) | 26 | n/d | n/d | n/d | no |

Espera de datos: fracción del tiempo del bucle de entrenamiento que el DataLoader tiene a la GPU ociosa (`ThroughputMonitor`). Por debajo de ~70 % de GPU se documenta y se considera más `num_workers` o un redimensionado previo (F3.4).

## 5. Colapsos

Ninguno.

## 6. Regla de decisión (F3.5) aplicada

Sin decisión: faltan semillas en ['A1', 'M1', 'M2'].

## 7. Anclaje en la literatura (F3.6)

Verificado en la fuente el 2026-09-17 (texto completo en ar5iv y resumen en Semantic Scholar).

| referencia | AUC-ROC declarado | sobre qué conjunto | modelos | resoluciones | metadatos | datos externos | TTA |
|:--|:--|:--|:--|:--|:--|:--|:--|
| Ha, Liu y Liu (2020), *Identifying Melanoma Images using EfficientNet Ensemble: Winning Solution to the SIIM-ISIC Melanoma Classification Challenge*, arXiv:2010.05351, <https://doi.org/10.48550/arXiv.2010.05351> | «0.9600 AUC on cross validation and 0.9490 AUC on private leaderboard» | validación cruzada de 5 pliegues sobre 2018+2019+2020 combinados, y leaderboard privado del reto | ensamble de **18 modelos** (EfficientNet B3–B7, SE-ResNeXt-101, ResNeSt-101) | 384, 448, 512, 576, 640, 768 y 896 | sí, en 4 de los 18 (sexo, edad, sitio anatómico, tamaño de imagen, n_images) | sí: ISIC 2018 y 2019 junto con 2020 | no se menciona en el texto consultado |
| Cassidy, Kendrick, Brodzicki, Jaworek-Korjakowska y Yap (2022), *Analysis of the ISIC image datasets: Usage, benchmarks and recommendations*, Medical Image Analysis 75, 102305, <https://doi.org/10.1016/j.media.2021.102305> | «an AUC of 0.80 for the best performing model» | conjunto de prueba de ISIC 2020, tras eliminar 14,310 duplicados del entrenamiento y balancear | modelos únicos (varias arquitecturas; el resumen no las nombra) | no indicado en el resumen | no | no («our aim was not to maximise network performance») | no |

Lectura: el ganador del reto es un ensamble de 18 redes grandes a resoluciones de hasta 896 px, con datos de tres años y metadatos en parte de los modelos. Un modelo único sin datos externos, sin TTA y sin metadatos, entrenado con la cuota gratuita de Kaggle, queda por diseño varios puntos por debajo; la referencia de modelo único de Cassidy et al. (0.80 en el test de ISIC 2020) es el punto de comparación más cercano a este pipeline, con la salvedad de que aquí se evalúa sobre un split de validación por paciente y no sobre el test del reto.

_El párrafo de situación se redacta con la configuración ganadora._

## 8. Modelo final para F4 (F3.7)

_pendiente_
