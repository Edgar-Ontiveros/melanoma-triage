# tf_efficientnetv2_s-224-s2-e153979

commit `e153979` · splits {'train': '701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c', 'val': 'b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1'}

| métrica | valor | IC 95 % (bootstrap por paciente) |
|:--|--:|:--|
| AUC-ROC | 0.818 | [0.760, 0.873] |
| AUPRC (primaria) | 0.169 | [0.108, 0.251] |
| Sensibilidad a especificidad 0.90 | 0.659 | [0.546, 0.755] |
| Sensibilidad a especificidad 0.95 | 0.466 | [0.360, 0.594] |
| Especificidad a sensibilidad 0.90 | 0.352 | [0.190, 0.635] |
| Especificidad a sensibilidad 0.95 | 0.191 | [0.120, 0.411] |
| Prevalencia (AUPRC de un clasificador aleatorio) | 0.0177 | |

Umbral: 0.0036 (política: sensibilidad>=0.9; la selección definitiva de umbral es de F4).

| | pred. benigno | pred. melanoma |
|:--|--:|--:|
| real benigno | 1718 | 3157 |
| real melanoma | 8 | 80 |

| métrica en el umbral | valor |
|:--|--:|
| Sensibilidad | 0.909 |
| Especificidad | 0.352 |
| VPP a la prevalencia observada (0.0177) | 0.025 |
| VPN a la prevalencia observada | 0.995 |
