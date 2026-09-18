# resnet50-224-s0-8da1f44

commit `8da1f44` · splits {'train': '701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c', 'val': 'b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1'}

| métrica | valor | IC 95 % (bootstrap por paciente) |
|:--|--:|:--|
| AUC-ROC | 0.719 | [0.672, 0.766] |
| AUPRC (primaria) | 0.032 | [0.024, 0.044] |
| Sensibilidad a especificidad 0.90 | 0.193 | [0.101, 0.288] |
| Sensibilidad a especificidad 0.95 | 0.023 | [0.000, 0.087] |
| Especificidad a sensibilidad 0.90 | 0.436 | [0.321, 0.568] |
| Especificidad a sensibilidad 0.95 | 0.326 | [0.140, 0.498] |
| Prevalencia (AUPRC de un clasificador aleatorio) | 0.0177 | |

Umbral: 0.5069 (política: sensibilidad>=0.9; la selección definitiva de umbral es de F4).

| | pred. benigno | pred. melanoma |
|:--|--:|--:|
| real benigno | 2124 | 2751 |
| real melanoma | 8 | 80 |

| métrica en el umbral | valor |
|:--|--:|
| Sensibilidad | 0.909 |
| Especificidad | 0.436 |
| VPP a la prevalencia observada (0.0177) | 0.028 |
| VPN a la prevalencia observada | 0.996 |

**COLAPSO** en épocas [0, 1, 2, 3, 4, 5]
