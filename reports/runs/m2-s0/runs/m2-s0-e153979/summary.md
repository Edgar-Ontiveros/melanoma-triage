# convnext_tiny-384-s0-e153979

commit `e153979` · splits {'train': '701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c', 'val': 'b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1'}

| métrica | valor | IC 95 % (bootstrap por paciente) |
|:--|--:|:--|
| AUC-ROC | 0.840 | [0.792, 0.883] |
| AUPRC (primaria) | 0.155 | [0.100, 0.238] |
| Sensibilidad a especificidad 0.90 | 0.614 | [0.500, 0.713] |
| Sensibilidad a especificidad 0.95 | 0.489 | [0.355, 0.594] |
| Especificidad a sensibilidad 0.90 | 0.525 | [0.293, 0.688] |
| Especificidad a sensibilidad 0.95 | 0.295 | [0.190, 0.609] |
| Prevalencia (AUPRC de un clasificador aleatorio) | 0.0177 | |

Umbral: 0.0005 (política: sensibilidad>=0.9; la selección definitiva de umbral es de F4).

| | pred. benigno | pred. melanoma |
|:--|--:|--:|
| real benigno | 2560 | 2315 |
| real melanoma | 8 | 80 |

| métrica en el umbral | valor |
|:--|--:|
| Sensibilidad | 0.909 |
| Especificidad | 0.525 |
| VPP a la prevalencia observada (0.0177) | 0.033 |
| VPN a la prevalencia observada | 0.997 |
