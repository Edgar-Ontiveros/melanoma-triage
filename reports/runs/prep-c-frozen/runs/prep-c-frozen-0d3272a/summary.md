# resnet50-224-s0-0d3272a

commit `0d3272a` · splits {'train': '701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c', 'val': 'b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1'}

| métrica | valor | IC 95 % (bootstrap por paciente) |
|:--|--:|:--|
| AUC-ROC | n/d | [n/d, n/d] |
| AUPRC (primaria) | n/d | [n/d, n/d] |
| Sensibilidad a especificidad 0.90 | n/d | [n/d, n/d] |
| Sensibilidad a especificidad 0.95 | n/d | [n/d, n/d] |
| Especificidad a sensibilidad 0.90 | n/d | [n/d, n/d] |
| Especificidad a sensibilidad 0.95 | n/d | [n/d, n/d] |
| Prevalencia (AUPRC de un clasificador aleatorio) | n/d | |

Umbral: 0.5000 (política: sensibilidad>=0.9; la selección definitiva de umbral es de F4).

| | pred. benigno | pred. melanoma |
|:--|--:|--:|
| real benigno | 0 | 4875 |
| real melanoma | 0 | 88 |

| métrica en el umbral | valor |
|:--|--:|
| Sensibilidad | 1.000 |
| Especificidad | 0.000 |
| VPP a la prevalencia observada (nan) | 0.018 |
| VPN a la prevalencia observada | n/d |

**COLAPSO** en épocas [0, 1, 2, 3, 4, 5]
